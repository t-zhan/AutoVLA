#!/bin/bash

REPO_ID="Qwen/Qwen2.5-VL-7B-Instruct"
LOCAL_DIR="./pretrained/Qwen2.5-VL-7B-Instruct"

mkdir -p "$LOCAL_DIR"

python tools/download/download_qwen.py \
    --repo_id "$REPO_ID" \
    --local_dir "$LOCAL_DIR" \