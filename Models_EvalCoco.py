from numpy.lib.function_base import append
from TFODPaths import get_paths_and_files

import os
 
import numpy as np
from matplotlib import pyplot as plt

from pycocotools.cocoeval import COCOeval
from pycocotools.coco import COCO

import sys
from io import StringIO

"""Load json file of ground truth and detection"""
# indicate custom model & desired checkpoint from training
CUSTOM_MODEL = 'my_centernet_hg104_1024_8'
TESTED_IMG_FOLDER = 'test_thresh0.5'

# get paths and files of custom model
paths, files = get_paths_and_files(CUSTOM_MODEL)
model_tested_path = os.path.join(paths['IMAGE_PATH'],'tested',CUSTOM_MODEL,TESTED_IMG_FOLDER)

annType = 'bbox'
#path_to_annotations_dir = os.path.join(path_to_coco_dir, 'annotations')
path_to_annotation = os.path.join(model_tested_path, 'COCO_annot.json')
path_to_results = os.path.join(model_tested_path, 'COCO_results.json')

cocoGt = COCO(path_to_annotation)
cocoDt = cocoGt.loadRes(path_to_results)

cocoEval = COCOeval(cocoGt, cocoDt, annType)

# set parameters as desired
#cocoEval.params.maxDets = [1,10,80]
    #  imgIds     - [all] N img ids to use for evaluation
    #  catIds     - [all] K cat ids to use for evaluation
    #  iouThrs    - [.5:.05:.95] T=10 IoU thresholds for evaluation
    #  recThrs    - [0:.01:1] R=101 recall thresholds for evaluation
    #  areaRng    - [...] A=4 object area ranges for evaluation
    #  maxDets    - [1 10 100] M=3 thresholds on max detections per image
    #  iouType    - ['segm'] set iouType to 'segm', 'bbox' or 'keypoints'
    #  iouType replaced the now DEPRECATED useSegm parameter.
    #  useCats    - [1] if true use category labels for evaluation

# evaluates detections on every image and every category
# concats results into evalImgs
# measures quality per image
evalImgs = cocoEval.evaluate()

# quality aggregated across entire dataset
# contains "precision" for every eval. setting; "recall" ~
eval = cocoEval.accumulate()

# compute 12 detection metrics based on eval structure
cocoEval.summarize()

#original_stdout = sys.stdout
#string_stdout = StringIO()
#sys.stdout = string_stdout
#sys.stdout = original_stdout
#mean_ap = cocoEval.stats[0].item()  # stats[0] records AP@[0.5:0.95]
#detail = string_stdout.getvalue()
#print(detail)

# NOTE: precision and recall==-1 for settings with no gt objects

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
plt.plot(x,pr_5, label="IoU@0.50")
plt.plot(x,pr_55, label="IoU@0.55")
plt.plot(x,pr_6, label="IoU@0.60")
plt.plot(x,pr_65, label="IoU@0.65")
plt.plot(x,pr_7, label="IoU@0.70")
plt.plot(x,pr_75, label="IoU@0.75")
plt.plot(x,pr_8, label="IoU@0.80")
plt.plot(x,pr_85, label="IoU@0.85")
plt.plot(x,pr_9, label="IoU@0.90")
plt.plot(x,pr_95, label="IoU@0.95")

plt.ylabel("Precision")
plt.xlabel("Recall")
#plt.legend()
#plt.legend(fontsize="small", loc="center left")
plt.legend(loc='lower center', bbox_to_anchor=(0.48, -0.3), ncol=5, fontsize = 'small')
plt.subplots_adjust(bottom=0.2)
plt.title((CUSTOM_MODEL+", 20 artificial images (mask JGIR)"))
plt.savefig(os.path.join(model_tested_path,"Precision_Recall_Curve.png"))
x=1