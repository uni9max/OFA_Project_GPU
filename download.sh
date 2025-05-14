#!/bin/bash
# Download required large files from Google Drive using gdown
mkdir -p checkpoints/caption
mkdir -p dataset/caption_data/cider_cached_tokens
gdown --id 1Dxbk-CMPrbB4BztEsYrTpxQDwLsg5HOr -O dataset/caption_data/caption_stage1_train.tsv
gdown --id 10SB1lQ9ENbrPaETCjzY6DJPm9GjYGNfS -O dataset/caption_data/caption_stage2_train.tsv
gdown --id 1JRtE-Ghi2Syu0RL-pcCKywcU8bBiResp -O dataset/caption_data/caption_test.tsv
gdown --id 1kyQTsNk8QsFGB2QF_oIj4C75j3IUQ6ag -O dataset/caption_data/caption_val.tsv
gdown --id 1DiHNM8MaqJnrQ6OrXLfuSnoTLpP0pEC1 -O dataset/caption_data/cider_cached_tokens/coco-train-words.p
gdown --id 1NQEzVWgSR_el4MsK-wuUb8o044iKhvij -O checkpoints/caption/caption_large_best_clean.pt
