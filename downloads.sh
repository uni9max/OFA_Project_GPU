#!/bin/bash

echo "Starting download of large files..."

# CHECKPOINT
gdown --fuzzy "https://drive.google.com/file/d/1NQEzVWgSR_el4MsK-wuUb8o044iKhvij/view?usp=drive_link"

# DATASETS
gdown --fuzzy "https://drive.google.com/file/d/1Dxbk-CMPrbB4BztEsYrTpxQDwLsg5HOr/view?usp=drive_link"
gdown --fuzzy "https://drive.google.com/file/d/10SB1lQ9ENbrPaETCjzY6DJPm9GjYGNfS/view?usp=drive_link"
gdown --fuzzy "https://drive.google.com/file/d/1JRtE-Ghi2Syu0RL-pcCKywcU8bBiResp/view?usp=drive_link"
gdown --fuzzy "https://drive.google.com/file/d/1kyQTsNk8QsFGB2QF_oIj4C75j3IUQ6ag/view?usp=drive_link"

# CACHED TOKEN
gdown --fuzzy "https://drive.google.com/file/d/1DiHNM8MaqJnrQ6OrXLfuSnoTLpP0pEC1/view?usp=drive_link"

echo "✅ All files downloaded successfully!"
