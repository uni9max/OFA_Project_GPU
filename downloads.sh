#!/bin/bash

echo "Starting download of large files..."

# CHECKPOINT
gdown --id 1ABCdefGHIJcaptionModelID -O checkpoints/caption/caption_large_best_clean.pt

# DATASETS
gdown --id 2XYZabcSTAGE1tsvID -O dataset/caption_data/caption_stage1_train.tsv
gdown --id 3LMNopqSTAGE2tsvID -O dataset/caption_data/caption_stage2_train.tsv
gdown --id 4JKLcapTESTtsvID -O dataset/caption_data/caption_test.tsv
gdown --id 5MNOcapVALtsvID -O dataset/caption_data/caption_val.tsv

# CACHED TOKEN
gdown --id 6PQRcocoTOKID -O dataset/caption_data/cider_cached_tokens/coco-train-words.p

echo "✅ All files downloaded successfully!"
