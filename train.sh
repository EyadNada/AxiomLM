#!/usr/bin/env bash
venv/bin/python brain/train_gpt2.py \
  --arch modern \
  --optimizer muon \
  --data_dir data/systems_shards \
  --batch_size 16384 \
  --save_interval 25 \
  --sample_prompt "import torch\nimport torch.nn as nn\n" \
  --resume checkpoints/model_latest.pt

