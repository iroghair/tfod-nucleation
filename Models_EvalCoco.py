from numpy.lib.function_base import append
from TFODPaths import get_paths_and_files

import os
import tensorflow as tf
from object_detection.utils import visualization_utils as viz_utils

import cv2 
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from object_detection.metrics import coco_tools

from pycocotools.cocoeval import COCOeval
from pycocotools.coco import COCO

import sys
from io import StringIO

"""Load json file of ground truth and detection"""
# indicate custom model & desired checkpoint from training
CUSTOM_MODEL = 'my_centernet_hg104_1024_5'
CUSTOM_CHECKPOINT = 'ckpt-21'
# max. allowed detections
max_detect = 500
# get paths and files of custom model
paths, files = get_paths_and_files(CUSTOM_MODEL)
model_tested_path = os.path.join(paths['IMAGE_PATH'],'tested',CUSTOM_MODEL,'test_thresh0.5')

annType = 'bbox'
#path_to_annotations_dir = os.path.join(path_to_coco_dir, 'annotations')
path_to_annotation = os.path.join(model_tested_path, 'COCO_annot.json')
path_to_results = os.path.join(model_tested_path, 'COCO_results.json')

cocoGt = COCO(path_to_annotation)
cocoDt = cocoGt.loadRes(path_to_results)

cocoEval = COCOeval(cocoGt, cocoDt, annType)
cocoEval.evaluate()
cocoEval.accumulate()

original_stdout = sys.stdout
string_stdout = StringIO()
sys.stdout = string_stdout
cocoEval.summarize()
sys.stdout = original_stdout

mean_ap = cocoEval.stats[0].item()  # stats[0] records AP@[0.5:0.95]
detail = string_stdout.getvalue()
# NOTE: precision and recall==-1 for settings with no gt objects
print(detail)

all_precision=cocoEval.eval['precision']
# COCO: IoU threshold range AP@[0.5:0.05:0.95]
pr_5 = all_precision[0, :, 0, 0, 2].tolist() # data for IoU@0.5
pr_55 = all_precision[1, :, 0, 0, 2].tolist() # data for IoU@
pr_6 = all_precision[2, :, 0, 0, 2].tolist() # data for IoU@
pr_65 = all_precision[3, :, 0, 0, 2].tolist() # data for IoU@
pr_7 = all_precision[4, :, 0, 0, 2].tolist() # data for IoU@0.7
pr_75 = all_precision[5, :, 0, 0, 2].tolist() # data for IoU@
pr_8 = all_precision[6, :, 0, 0, 2].tolist() # data for IoU@
pr_85 = all_precision[7, :, 0, 0, 2].tolist() # data for IoU@
pr_9 = all_precision[8, :, 0, 0, 2].tolist() # data for IoU@
pr_95 = all_precision[9, :, 0, 0, 2].tolist() # data for IoU@
x = np.arange(0, 1.01, 0.01)
plt.plot(x,pr_5, label="IoU@0.5")
plt.plot(x,pr_55, label="IoU@0.55")
plt.plot(x,pr_6, label="IoU@0.6")
plt.plot(x,pr_65, label="IoU@0.65")
plt.plot(x,pr_7, label="IoU@0.7")
plt.plot(x,pr_75, label="IoU@0.75")
plt.plot(x,pr_8, label="IoU@0.8")
plt.plot(x,pr_85, label="IoU@0.85")
plt.plot(x,pr_9, label="IoU@0.9")
plt.plot(x,pr_95, label="IoU@0.95")

plt.ylabel("Precision")
plt.xlabel("Recall")
plt.legend()
plt.title((CUSTOM_MODEL+", img: 20 artificial, mask JGIR"))
plt.savefig(os.path.join(model_tested_path,"TestPRC_allimgs_absPixel.png"))
x=1