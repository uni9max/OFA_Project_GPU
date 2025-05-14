# evaluate_cider.py

import sys
import json
from utils.cider.pyciderevalcap.eval import COCOEvalCap
from utils.cider.pyciderevalcap.coco import COCO

# Paths to your files
ref_path = "results/caption/captions_val2014.json"
hyp_path = "results/caption/test_predict.json"

# Load reference and prediction
coco = COCO(ref_path)
cocoRes = coco.loadRes(hyp_path)

# Run evaluation
cocoEval = COCOEvalCap(coco, cocoRes)
cocoEval.evaluate()

# Print only CIDEr
print(f"CIDEr: {cocoEval.eval['CIDEr']:.4f}")
