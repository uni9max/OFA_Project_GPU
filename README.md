# OFA_Project_GPU
Modified Version of OFA (Image Captioning). CIDEr + SPICE metrics are implemented.

## 📦 Download Required Files

Before running training or evaluation, download necessary files with:

```bash
bash download.sh

Make sure Python + gdown are installed:
pip install gdown

Each file can be placed accordingly:
- checkpoints/caption/caption_large_best_clean.pt
- dataset/caption_data/caption_stage1_train.tsv
- dataset/caption_data/caption_stage2_train.tsv
- dataset/caption_data/caption_test.tsv
- dataset/caption_data/caption_val.tsv
- dataset/caption_data/cider_cached_tokens/coco-train-words.p
