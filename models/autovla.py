import torch
import os
from tqdm import tqdm
from typing import Dict, Any
import pytorch_lightning as pl
from pathlib import Path
import torch.nn.functional as F
import numpy as np
from typing import List
from torch.distributed.fsdp import StateDictType
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from models.action_tokenizer import ActionTokenizer
from transformers.modeling_outputs import CausalLMOutputWithPast
from models.utils.score import PDM_Reward, TrajectorySampling, Trajectory


class GRPOAutoVLA(pl.LightningModule):
    def __init__(self, config: dict, inference=False):
        super().__init__()
        self.cfg = config
        self.use_cot = config['model']['use_cot']
        self.save_hyperparameters()

        # Load trajectory sampling from config or use default
        traj_conf = config['model']['trajectory']
        self.trajectory_sampling = TrajectorySampling(
            num_poses=traj_conf['num_poses'],
            interval_length=traj_conf['interval_length']
        )
        
        # Load token configs
        token_conf = config['model']['tokens']
        self.action_start_id = token_conf['action_start_id']
        self.assistant_id = torch.tensor(token_conf['assistant_id'])

        # Training model (wrapped by Lightning FSDPStrategy)
        self.autovla = AutoVLA(config)

        self.autovla.train()
        self._train_vision_backbone = config['model']['train_vision_backbone']
        self._train_llm_backbone = config['model']['train_lm_backbone']

        # online reference model.
        if not inference:
            self.reference_model = AutoVLA(config, inference=True)
            state_dict = torch.load(config['model']['sft_model_path'])["state_dict"]
            state_dict = {k.replace("autovla.", "").replace("drivevla.", ""): v for k, v in state_dict.items()}
            self.reference_model.load_state_dict(state_dict, strict=False)
            self.reference_model.eval()  
            print(f"Using online reference model from {config['model']['sft_model_path']}")

        # sample generation config
        sample_conf = config['training']['sample']
        self._sample_generation_temperature = {
            "max_length": sample_conf['max_length'],
            "temperature": sample_conf['temperature'],
            "top_k": sample_conf['top_k'],
            "top_p": sample_conf['top_p'],
        }

        # reward function
        self.train_critic = PDM_Reward(Path(config['data']['train']['metric_cache_path']))
        self.val_critic = PDM_Reward(Path(config['data']['val']['metric_cache_path']))

        # sliding window for training reward
        if not inference:
            self.window_size = config['rl']['reward'].get("sliding_window_size", 100)
            self.register_buffer("training_reward_buffer", torch.zeros(self.window_size))
            self.register_buffer("sliding_idx",   torch.zeros(1, dtype=torch.long))
            self.register_buffer("window_count",  torch.zeros(1, dtype=torch.long))

    def training_step(self, batch):
        # Generate a sample from the model.
        self.autovla.train()
        with torch.no_grad():
            sample = self.generate_sample(
                batch, model=self.autovla, device=next(self.parameters()).device)
        
            # Compute the reward for the generated sample.
            reward = self.reward_function(sample)
            reward_scale = self.cfg['rl']['reward'].get("scale", 1.0)
            reward = reward * reward_scale
            
            # Normalize the rewards to compute the advantage.
            groupped_rewards = self.all_gather(reward)
            print(groupped_rewards)
            advantage = (reward - groupped_rewards.mean()) / (groupped_rewards.std() + 1e-4)

        # Compute the per-token log probabilities.
        per_token_logps = self.get_per_token_logps(
            self.autovla.vlm, 
            sample['input_ids'], 
            sample['attention_mask'], 
            sample['pixel_values_videos'], 
            sample['video_grid_thw']
        )
        # Get rid of the prompt (-1 because of the shift done in get_per_token_logps)
        per_token_logps = per_token_logps[:, sample['prompt_length']-1:]
        completion_mask = sample['completion_mask']

        # reference model
        with torch.no_grad():
            ref_per_token_logps = self.get_per_token_logps(
                self.reference_model.vlm, 
                sample["input_ids"], 
                sample["attention_mask"], 
                sample["pixel_values_videos"], 
                sample["video_grid_thw"]
            )
            ref_per_token_logps = ref_per_token_logps[:, sample["prompt_length"]-1:]

        # Compute the policy loss
        per_policy_loss = \
            torch.exp(per_token_logps - per_token_logps.detach()) * advantage.unsqueeze(-1)

        # Compute the kl loss
        kl_beta = self.cfg['rl'].get("kl_beta", 0.0)
        per_token_kl = \
            torch.exp(ref_per_token_logps - per_token_logps) - (ref_per_token_logps - per_token_logps) - 1
        per_kl_loss = kl_beta * per_token_kl

        per_token_loss = -(per_policy_loss - per_kl_loss)
        loss = ((per_token_loss * completion_mask).sum(dim=1) / completion_mask.sum(dim=1)).mean()

        # Log metrics
        self.log("loss", loss, sync_dist=True, prog_bar=True)
        per_kl_loss = ((per_kl_loss * completion_mask).sum(dim=1) / completion_mask.sum(dim=1)).mean()
        self.log("kl_divergence", per_kl_loss, sync_dist=True)

        # record training reward
        self.training_buffer_record(reward.mean())
        return loss
    
    def training_buffer_record(self, step_reward):
        idx = self.sliding_idx.item()
        self.training_reward_buffer[idx] = step_reward

        new_idx = (idx + 1) % self.window_size
        self.sliding_idx.fill_(new_idx)
        new_count = min(self.window_count.item() + 1, self.window_size)
        self.window_count.fill_(new_count)

        if new_count >= self.window_size:
            sliding_avg = self.training_reward_buffer.mean()
            self.log(
                "avg_train_reward",
                sliding_avg,
                sync_dist=False, 
                prog_bar=True
            )

    def on_after_backward(self):
        total_norm = 0.0
        for p in self.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** 0.5
        self.log("grad_norm", total_norm, sync_dist=True)
    
    def reward_function(self, sample):
        device = next(self.parameters()).device

        # Add pdm score for nuplan scenario
        reward = self.train_critic.rl_pdm_score(sample['trajectory'], sample['token'])
        reward = torch.tensor(reward).to(device)

        # Add chain-of-thought penalty (if "need cot" is found in the generated text).
        if self.use_cot:
            cot_conf = self.cfg['rl']['cot_penalty']
            cot_penalty_coef = cot_conf['coef']
            center = cot_conf['center']
            cot_penalty_weight = cot_conf['weight']

            cot_penalties = torch.stack([
                torch.sigmoid(torch.tensor(
                    (len(text) - center) * cot_penalty_coef,
                    device=device,
                    dtype=reward.dtype
                ))
                if "complex scenario" in text.lower() else torch.tensor(
                    0.0, device=device, dtype=reward.dtype
                )
                for text in sample['completion_texts']
            ])
            reward = reward - cot_penalty_weight * cot_penalties
        else:
            cot_penalties = torch.tensor(0.0, device=device, dtype=reward.dtype)

        self.log("train_reward", reward, sync_dist=True, prog_bar=True, on_step=True, on_epoch=False)
        self.log("cot_penalty", cot_penalties.mean(), sync_dist=True, prog_bar=True, on_step=True, on_epoch=False)

        return reward
    
    def get_per_token_logps(self, model, input_ids, attention_mask, pixel_values_videos, video_grid_thw):
        # Get the per-token log probabilities for the completions for the model and the reference model
        logits = model(input_ids, attention_mask=attention_mask, 
                       pixel_values_videos=pixel_values_videos, 
                       video_grid_thw=video_grid_thw).logits  # (B, L, V)
        
        logits = logits[:, :-1, :]  # (B, L-1, V), exclude the last logit: it corresponds to the next token pred
        input_ids = input_ids[:, 1:]  # (B, L-1), exclude the first input ID since we don't have logits for it

        # Compute the log probabilities for the input tokens. Use a loop to reduce memory peak.
        log_probs = torch.log_softmax(logits, dim=-1)  # (B, L-1, V)
        per_token_logps = log_probs.gather(2, input_ids.unsqueeze(-1)).squeeze(-1)  # (B, L-1)
        return per_token_logps

    def generate_sample(self, data, model, device):

        # Get the model inputs
        inputs = model.get_prompt(data['input_features'])
        model_inputs = {k: v.to(device) for k, v in inputs.items() if isinstance(v, torch.Tensor)}

        # set seed
        torch.manual_seed(int(str(device).split(':')[-1]))

        # Generate completion
        with torch.no_grad():
            prompt_completion_ids = model.vlm.generate(
                **model_inputs,
                do_sample=True,
                max_length=self._sample_generation_temperature['max_length'],
                temperature=self._sample_generation_temperature['temperature'],
                top_k=self._sample_generation_temperature['top_k'],
                top_p=self._sample_generation_temperature['top_p'],
            )

            prompt_length = inputs.input_ids.size(1)
            prompt_mask = model_inputs['attention_mask']
            completion_ids = prompt_completion_ids[:, prompt_length:]

            # Extract action tokens and trajectory (! batch size = 1)
            actions_tokens = completion_ids[0][completion_ids[0] >= self.action_start_id]

            if len(actions_tokens) > self.trajectory_sampling.num_poses:
                actions_tokens = actions_tokens[:self.trajectory_sampling.num_poses]
            elif len(actions_tokens) < self.trajectory_sampling.num_poses:
                actions_tokens = torch.cat([actions_tokens, torch.zeros(self.trajectory_sampling.num_poses - len(actions_tokens)).to(device)])
                actions_tokens = actions_tokens.long()
            else:
                pass

            trajectory = self.autovla.action_tokenizer.decode_token_ids_to_trajectory(actions_tokens.cpu())[0, 1:]
            trajectory = Trajectory(trajectory.cpu().numpy(), self.trajectory_sampling)

            # Create completion mask
            is_eos = completion_ids == model.processor.tokenizer.eos_token_id
            eos_idx = torch.full((is_eos.size(0),), is_eos.size(1), dtype=torch.long, device=device)
            eos_idx[is_eos.any(dim=1)] = is_eos.int().argmax(dim=1)[is_eos.any(dim=1)]
            sequence_indices = torch.arange(is_eos.size(1), device=device).expand(is_eos.size(0), -1)
            completion_mask = (sequence_indices <= eos_idx.unsqueeze(1)).int()

            # Concatenate prompt_mask with completion_mask for logit computation
            attention_mask = torch.cat([prompt_mask, completion_mask], dim=1) 

            completion_texts = model.processor.batch_decode(completion_ids)

            # Create outputs
            outputs = {'trajectory': trajectory, 
                       'token': data['token'], 
                       'completion_texts': completion_texts,
                       'prompt_length': prompt_length,
                       'input_ids': prompt_completion_ids, 
                       'completion_ids': completion_ids,
                       'attention_mask': attention_mask,
                       'completion_mask': completion_mask,
                       'pixel_values_videos': model_inputs['pixel_values_videos'], 
                       'video_grid_thw': model_inputs['video_grid_thw'],
                        }
        
        # clean up
        torch.cuda.empty_cache()

        return outputs
    
    def configure_optimizers(self):
        if not self._train_vision_backbone:
            for param in self.autovla.vlm.visual.parameters():
                param.requires_grad = False

        if not self._train_llm_backbone:
            for param in self.autovla.vlm.model.parameters():
                param.requires_grad = False

        params_to_update = []
        for param in self.autovla.vlm.parameters():
            if param.requires_grad == True:
                params_to_update.append(param)

        assert len(params_to_update) > 0, 'No parameters to update'

        lr = float(self.cfg['training']['learning_rate'])
        wd = float(self.cfg['training'].get('weight_decay', 0.0))
        optimizer = torch.optim.AdamW(
            params_to_update,
            lr=lr,
            weight_decay=wd
        )

        return optimizer
    
    def configure_gradient_clipping(self, optimizer, gradient_clip_val, gradient_clip_algorithm):
        # Filter out parameters with no gradient to avoid empty tensor lists
        params_with_grad = [p for p in self.parameters() if p.grad is not None]
        if params_with_grad:
            torch.nn.utils.clip_grad_value_(params_with_grad, clip_value=gradient_clip_val)

    def on_save_checkpoint(self, checkpoint: dict):
        # only save main model
        sd = checkpoint.get("state_dict", {})
        for k in list(sd):
            if k.startswith("reference_model."):
                sd.pop(k)

class SFTAutoVLA(pl.LightningModule):
    def __init__(self, config: dict):
        super().__init__()
        self.cfg = config
        self.save_hyperparameters()

        self.autovla = AutoVLA(config)
        self.autovla.train()

        self._train_vision_backbone = config['model']['train_vision_backbone']
        self._train_llm_backbone = config['model']['train_lm_backbone']

    def training_step(self, batch):
        hascot = batch['has_cot']
        gt_trajectory = batch["gt_trajectory"]
        gt_action = batch["gt_action"]
        output = self.autovla(batch)
        loss = output.loss

        # === Add additional loss on action tokens ===
        # output.logits shape: (B, T, V), labels shape: (B, T)
        logits = output.logits
        vocab_size = logits.size(-1)
        # Flatten logits and labels for token-wise loss
        labels = batch['labels']
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        logits_flat = shift_logits.view(-1, vocab_size)
        labels_flat = shift_labels.view(-1)
        # Identify action token positions
        action_mask = (labels_flat >= self.autovla.action_start_id)  # shape: (B*T,)
        # Compute token-wise cross-entropy loss
        ce_loss_all = F.cross_entropy(logits_flat, labels_flat, reduction='none')  # shape: (B*T,)
        # Extract loss for action tokens
        action_loss = ce_loss_all[action_mask]
        # Add to total loss with optional weighting factor
        if action_loss.numel() > 0:
            action_loss = action_loss.mean()

        # # add more penalty for CoT reasoning data
        if hascot[0] == True:
            # print("add more penalty for CoT reasoning data")
            loss = loss * 40
            loss = loss + action_loss

        self.log("train_loss", loss.item(),
                 batch_size=gt_action.shape[0],
                 sync_dist=True,
                 prog_bar=True)
        
        
        return loss
    
    def validation_step(self, batch):
        gt_trajectory = batch["gt_trajectory"]
        gt_action = batch["gt_action"]

        output = self.autovla(batch)
        loss = output.loss
        self.log("val_loss", loss.item(),
                 batch_size=gt_action.shape[0],
                 sync_dist=True, prog_bar=True)
        
        return loss
    
    def configure_optimizers(self):
        if not self._train_vision_backbone:
            for param in self.autovla.vlm.visual.parameters():
                param.requires_grad = False

        if not self._train_llm_backbone:
            for param in self.autovla.vlm.model.parameters():
                param.requires_grad = False

        params_to_update = []
        for param in self.autovla.vlm.parameters():
            if param.requires_grad == True:
                params_to_update.append(param)

        assert len(params_to_update) > 0, 'No parameters to update'

        optimizer = torch.optim.AdamW(
            params_to_update,
            lr=self.cfg['training']['learning_rate'],
            weight_decay=self.cfg['training'].get('weight_decay', 0.0)
        )
        lr_warmpup_step = self.cfg['training']['lr_warmup_step']
        lr_step_freq = self.cfg['training']['lr_step_frequency']
        lr_step_gamma = self.cfg['training']['lr_step_gamma']

        def lr_update(step, warmup_step, step_size, gamma):
            if step < warmup_step:
                # warm up lr
                lr_scale = 1 - (warmup_step - step) / warmup_step * 0.95
            else:
                n = (step - warmup_step) // step_size
                lr_scale = gamma ** n

            if lr_scale < 1e-2:
                lr_scale = 1e-2
            elif lr_scale > 1:
                lr_scale = 1

            return lr_scale
        
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=lambda step: lr_update(
                step,
                lr_warmpup_step,
                lr_step_freq,
                lr_step_gamma,
            )
        )
        return [optimizer], [{"scheduler": scheduler, "interval": "step"}]
    
    @torch.no_grad()
    def calculate_metrics(self, logits, labels, gt_trajectory):
        # Find start index for ground truth sequence
        gt_start_idx = self.find_assistant_start_idx(labels[0])
        gt_tokens = labels[0, gt_start_idx+1:] # shifted
        pred_tokens = logits[0, gt_start_idx:-1].argmax(dim=-1)

        # Find action tokens in ground truth and predicted sequences
        gt_action_idx = gt_tokens >= self.autovla.action_start_id
        pred_action_idx = pred_tokens >= self.autovla.action_start_id

        if len(pred_tokens[pred_action_idx]) != len(gt_tokens[gt_action_idx]):
            pred_action_idx = gt_action_idx
            
        gt_action_tokens = gt_tokens[gt_action_idx]
        pred_action_tokens = pred_tokens[pred_action_idx]

        # Decode predicted trajectory
        # pred_trajectory = self.autovla.action_tokenizer.decode_token_ids_to_trajectory(pred_action_tokens.cpu())
        # action_acc = (pred_action_tokens == gt_action_tokens).float().mean()
        # traj_mse = torch.norm(pred_trajectory[0, 1:, :2] - gt_trajectory[0].cpu(), dim=-1).mean()
        # traj_mse = traj_mse.to(logits.device)

        # return {
        #     'action_acc': action_acc,
        #     'traj_mse': traj_mse
        # }
    
    @staticmethod
    def find_assistant_start_idx(labels):
        assistant_id = torch.tensor(ASSISTANT_ID).to(labels.device)
        
        for j in range(len(labels) - len(assistant_id) + 1):
            if torch.equal(labels[j:j + len(assistant_id)], assistant_id):
                start_idx = j
                break

        return start_idx


class AutoVLA(torch.nn.Module):
    def __init__(self, config, inference=False, device='cpu'):
        super().__init__()
        self.device = device

        model_path = config['model']['pretrained_model_path']
        self.vlm = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map=device
        )
        self.processor = AutoProcessor.from_pretrained(model_path)
        self.action_tokenizer = ActionTokenizer(self.processor.tokenizer, 
                                                model_config=config['model'])
        self.vlm.resize_token_embeddings(len(self.processor.tokenizer))

        self.video_conf = config['model']['video']
        self.action_start_id = config['model']['tokens']['action_start_id']

        self.use_cot = config['model']['use_cot']
        self.gen_conf = config['inference']['sample']

    def predict(self, input_features):
        inputs = self.get_prompt(input_features)
        model_inputs = {k: v.to(self.device) for k, v in inputs.items() if isinstance(v, torch.Tensor)}

        outputs = self.vlm.generate(
            **model_inputs,
            max_length=self.gen_conf['max_length'],
            do_sample=True,
            temperature=self.gen_conf['temperature'],
            top_k=self.gen_conf['top_k'],
            top_p=self.gen_conf['top_p'],
        )

        outputs_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, outputs)
        ]

        outputs_trimmed = outputs_trimmed[0][:-1].cpu() # remove end token
        cot_results = self.processor.decode(outputs_trimmed)
        # if 'Chain-of-Thought is not needed' not in self.processor.decode(outputs_trimmed):
        #     print(self.processor.decode(outputs_trimmed))
        #     print("has cot")
        # else:
        #     print(self.processor.decode(outputs_trimmed))
        #     print("no cot")
        actions_tokens = outputs_trimmed[outputs_trimmed >= self.action_start_id]

        trajectory = self.action_tokenizer.decode_token_ids_to_trajectory(actions_tokens)[0, 1:]

        return trajectory, cot_results
    
    def get_prompt(self, input_features, image_mode="video"):
        # image sensor
        images = input_features['images']

        min_pixels = self.video_conf.get("min_pixels", 28 * 28 * 128)
        max_pixels = self.video_conf.get("max_pixels", 28 * 28 * 128)

        camera_images = {}
        
        # List of camera types to load
        camera_types = ['front_camera', 'front_left_camera', 'front_right_camera']
        
        if input_features['sensor_data_path']:
            for camera_type in camera_types:
                camera_images[camera_type] = []
                for i in range(4):
                    img = images[camera_type][i]
                    camera_images[camera_type].append(
                        os.path.join(input_features['sensor_data_path'], img))

        # Assign to individual variables for message formatting
        front_camera_1, front_camera_2, front_camera_3, front_camera_4 = camera_images['front_camera']
        front_left_camera_1, front_left_camera_2, front_left_camera_3, front_left_camera_4 = camera_images['front_left_camera']
        front_right_camera_1, front_right_camera_2, front_right_camera_3, front_right_camera_4 = camera_images['front_right_camera']


        # vehicle state
        velocity = input_features["vehicle_velocity"]

        if isinstance(velocity, list) or isinstance(velocity, np.ndarray):
            velocity_x = velocity[0]
            velocity_y = velocity[1]
            velocity = np.sqrt(velocity_x**2 + velocity_y**2)
    
        acceleration = input_features["vehicle_acceleration"]
        if isinstance(acceleration, list) or isinstance(acceleration, np.ndarray):
            acceleration_x = acceleration[0]
            acceleration_y = acceleration[1]
            acceleration = np.sqrt(acceleration_x**2 + acceleration_y**2)

        instruction = input_features["driving_command"].lower()
    
        user_content = [
            {
                "type": "text",
                "text": (
                    "The autonomous vehicle is equipped with three cameras mounted at the front, left, and right, enabling a comprehensive perception of the surrounding environment."
                )
            },
            {
                "type": "text",
                "text": "The first video presents the front view of the vehicle, comprising four sequential frames sampled at 2 Hz."
            },
            {
                "type": "video",
                "min_pixels": min_pixels,
                "max_pixels": max_pixels,
                "video": [
                    f"file://{front_camera_1}",
                    f"file://{front_camera_2}",
                    f"file://{front_camera_3}",
                    f"file://{front_camera_4}",
                ]
            },
            {
                "type": "text",
                "text": "The second video presents the front-left view of the vehicle, comprising four sequential frames sampled at 2 Hz."
            },
            {
                "type": "video",
                "min_pixels": min_pixels,
                "max_pixels": max_pixels,
                "video": [
                    f"file://{front_left_camera_1}",
                    f"file://{front_left_camera_2}",
                    f"file://{front_left_camera_3}",
                    f"file://{front_left_camera_4}",
                ]
            },
            {
                "type": "text",
                "text": "The third video presents the front-right view of the vehicle, comprising four sequential frames sampled at 2 Hz."
            },
            {
                "type": "video",
                "min_pixels": min_pixels,
                "max_pixels": max_pixels,
                "video": [
                    f"file://{front_right_camera_1}",
                    f"file://{front_right_camera_2}",
                    f"file://{front_right_camera_3}",
                    f"file://{front_right_camera_4}",
                ]
            },
            {
                "type": "text",
                "text": (
                    f"The current velocity of the vehicle is {velocity:.3f} m/s, and the current acceleration is {acceleration:.3f} m/s². "
                    f"The driving instruction is: {instruction}. Based on this information, plan the action trajectory for the autonomous vehicle over the next five seconds."
                )
            },
        ]

        if self.use_cot:
            messages = [
                {   
                    "role": "system",
                    "content": [
                        {
                            "type": "text",
                            "text":
                            "You are an Advanced Driver Assistance and Full Self-Driving System. "
                            "You will receive visual observations from the ego vehicle’s cameras and dynamic information about the vehicle’s current state. "
                            "Your task is to predict the optimal driving action for the next five seconds.\n\n"
                            "First, carefully analyze the surrounding environment by considering traffic lights, the movements of other vehicles and pedestrians, lane markings, and any other relevant factors.\n\n"
                            "If necessary, use step-by-step reasoning (Chain-of-Thought) to arrive at the best driving action. Otherwise, you may directly predict the final driving action.\n\n"
                            # "Structure your reasoning as follows:\n"
                            # "1. **Scene Analysis**: Describe the traffic situation, including relevant environmental cues such as traffic lights, lane markings, and the behaviors of surrounding vehicles or pedestrians.\n"
                            # "2. **Identification of Critical Objects**: Identify two to three critical road users or obstacles, specifying their relative positions to the ego vehicle.\n"
                            # "3. **Prediction of Critical Object Behavior**: Predict the potential movements of the identified critical objects.\n"
                            # "4. **Ego Vehicle Intent Reasoning**: Based on the observed environment and current vehicle state, reason about the desired intent of the ego vehicle.\n"
                            # "5. **Final Action Decision**: Select one lateral action and one longitudinal action:\n"
                            # "- **Lateral actions** (choose exactly one): [move forward, turn left, change lane to left, turn right, change lane to right]\n"
                            # "- **Longitudinal actions** (choose exactly one): [stop, deceleration to zero, maintain constant speed, quick deceleration, deceleration, quick acceleration, acceleration]\n\n"
                            "Present the final action clearly after your reasoning steps."
                        }
                    ]
                },

                {
                    "role": "user",
                    "content": user_content
                },


            ]
        else:
            messages = [
                {   
                    "role": "system",
                    "content": [
                        {
                            "type": "text",
                            "text":
                            "You are an Advanced Driver Assistance and Full Self-Driving System. "
                            "You will be provided with video observations from the ego vehicle’s surrounding cameras, along with the vehicle’s current dynamic states. "
                            "Your task is to predict the most appropriate driving action for the next five seconds."
                        }
                    ]
                },
                {
                    "role": "user",
                    "content": user_content
                },
            ]

        image_inputs, video_inputs = process_vision_info(messages)
        
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, add_vision_id=True
        )

        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )

        return inputs
    
    def forward(self, inputs):
        inputs.pop('gt_trajectory')
        inputs.pop('gt_action')
        inputs.pop('has_cot')
        outputs: CausalLMOutputWithPast = self.vlm(**inputs)

        return outputs