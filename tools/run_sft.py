import sys
from pathlib import Path

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "navsim"))

import yaml
import torch
import argparse
import functools
import os
import pytorch_lightning as pl

import swanlab
from swanlab.integration.pytorch_lightning import SwanLabLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning import seed_everything
from pytorch_lightning.strategies import FSDPStrategy

from torch.distributed.fsdp import MixedPrecision
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from torch.distributed.fsdp import BackwardPrefetch
from torch.utils.data import DataLoader

from dataset_utils.sft_dataset import SFTDataset, DataCollator
from models.autovla import SFTAutoVLA
from transformers import AutoProcessor
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLDecoderLayer
import datetime

torch.set_float32_matmul_precision('high')


def load_config(file_path):
    with open(file_path, 'r') as file:
        config = yaml.safe_load(file)
    return config


if __name__ == "__main__":
    # Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume training from")
    args = parser.parse_args()
    seed_everything(args.seed)

    # Load configuration
    config = load_config(f"./config/{args.config}.yaml")

    # Model, dataset, and dataloader
    processor = AutoProcessor.from_pretrained(config['model']['pretrained_model_path'], use_fast=True)
    
    # Get using_cot setting from config (default to True if not specified)
    using_cot = config['model']['use_cot']
    

    train_dataset = SFTDataset(config['data']['train'], config['model'], processor, using_cot=using_cot)
        
    # Randomly sample from training set if train_sample_size is specified
    train_sample_size = config['training']['train_sample_size']
    if train_sample_size is not None and len(train_dataset) > train_sample_size:
        indices = torch.randperm(len(train_dataset))[:train_sample_size]
        train_dataset = torch.utils.data.Subset(train_dataset, indices)
    else:
        print("no sampling")
        
    val_dataset = SFTDataset(config['data']['val'], config['model'], processor, using_cot=using_cot)

    model = SFTAutoVLA(config)
    model.autovla.vlm.model.gradient_checkpointing_enable()

    # Resume checkpoint (full state: weights, optimizer, LR scheduler, epoch)
    ckpt_path = None
    swanlab_id = None
    if args.resume:
        ckpt = torch.load(args.resume, map_location='cpu')
        if 'optimizer_states' in ckpt:
            ckpt_path = args.resume   # full resume via Lightning
            print(f"[Resume] Full resume from {args.resume} (epoch={ckpt.get('epoch', '?')})")
        else:
            model.load_state_dict(ckpt['state_dict'], strict=False)
            print(f"[Resume] Weight-only warm-start from {args.resume} (optimizer starts fresh)")
        # Extract old SwanLab run ID from checkpoint path (e.g. run-xxx/files/ckpt -> run-xxx -> id)
        run_dir_name = Path(args.resume).resolve().parent.parent.name
        if run_dir_name.startswith("run-"):
            swanlab_id = run_dir_name.split("-")[-1]
            print(f"[Resume] Will continue SwanLab run {swanlab_id}")
    
    # Create data collator with config parameters
    data_collator = DataCollator(
        processor=processor,
        ignore_index=config['model']['tokens']['ignore_index'],
        assistant_id=config['model']['tokens']['assistant_id']
    )
    
    train_data = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        collate_fn=data_collator,
        num_workers=config['training']['num_workers'],
        shuffle=True,
    )

    val_data = DataLoader(
        val_dataset,
        batch_size=config['inference']['batch_size'],
        collate_fn=data_collator,
        num_workers=config['inference']['num_workers'],
        shuffle=False,
    )    

    # Training
    wrap_policy = functools.partial(
        transformer_auto_wrap_policy,
        transformer_layer_cls={
            Qwen2_5_VLDecoderLayer
        },
    )

    current_date = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    # Pre-init SwanLab on rank 0 to get the run directory for checkpoint placement.
    # FSDP spawns subprocesses that re-execute this script; guard against multi-init.
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    if local_rank == 0:
        run = swanlab.init(
            project="autovla",
            experiment_name="sft-" + current_date,
            logdir="runs/sft",
            id=swanlab_id,
            resume="allow",
        )
        run_dir = run.public.run_dir  # e.g. runs/sft/run-20260602_015001-xxxxx
    else:
        run_dir = "runs/sft"  # unused on non-rank-0, only rank 0 writes checkpoints
    
    trainer = pl.Trainer(
        num_nodes=1,
        max_epochs=config['training']['epochs'],
        accelerator="gpu",
        devices='auto',
        log_every_n_steps=1,
        accumulate_grad_batches=config['training']['accumulate_grad_batches'],
        strategy=FSDPStrategy(
            auto_wrap_policy=wrap_policy,
            cpu_offload=True,
            # Mixed precision training
            mixed_precision=MixedPrecision(
                param_dtype=torch.bfloat16,
                reduce_dtype=torch.bfloat16,
                buffer_dtype=torch.bfloat16
            ),
            # sharding strategy
            sharding_strategy='FULL_SHARD',
            # prefetching backward computation
            backward_prefetch = BackwardPrefetch.BACKWARD_PRE,
            # save state dict type
            state_dict_type="full", # can be full or sharded
            limit_all_gathers=True, # limit all_gathers to save memory
        ),
        callbacks=[
            ModelCheckpoint(
                monitor="val_loss",
                mode="min",
                save_top_k=3,
                dirpath=os.path.join(run_dir, "files"),
                filename="epoch={epoch}-loss={val_loss:.4f}",
                auto_insert_metric_name=False,
                save_weights_only=False,
                every_n_epochs=1,
            ),
            EarlyStopping(monitor="val_loss", patience=10, mode="min"),
            LearningRateMonitor(logging_interval="step"),
        ],
        gradient_clip_algorithm = 'value',
        gradient_clip_val = 1.0,

        logger=SwanLabLogger(project="autovla", experiment_name="sft-" + current_date, logdir="runs/sft", id=swanlab_id),
        enable_model_summary=True,

        # limit_val_batches=0.001
    )
    torch.cuda.empty_cache()
    trainer.fit(model, train_dataloaders=train_data, val_dataloaders=val_data, ckpt_path=ckpt_path)