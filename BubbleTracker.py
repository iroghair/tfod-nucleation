from TFODPaths import get_paths_and_files
import os
import json
import pandas as pd
import motrackers.centroid_kf_tracker

"""Load json file of ground truth and detection"""
# indicate custom model & desired checkpoint from training
CUSTOM_MODEL = 'my_centernet_hg104_1024_7'
TESTED_IMG_FOLDER = 'test_thresh0.5'

# get paths and files of custom model
paths, files = get_paths_and_files(CUSTOM_MODEL)
model_tested_path = os.path.join(paths['IMAGE_PATH'],'tested',CUSTOM_MODEL,TESTED_IMG_FOLDER)

path_to_results = os.path.join(model_tested_path, 'COCO_results.json')
with open(path_to_results) as json_file:
    json_data = json.load(json_file)
    print(json_data)
df = pd.DataFrame(json_data)
dict = {}
for img in df.image_id.unique():
    dict[img] = df[df.image_id==img]

# bbox: x, y, width, height
box_dict = {}
for key in dict:
    d = dict[key]
    boxes = pd.DataFrame(d['bbox'].to_list(),columns=['x','y','width','height'])
    #new_d = pd.concat([d, boxes], axis=1)
    box_dict[key] = boxes.to_numpy()

iname_sorted = sorted(box_dict.keys())
idx = 0
match_dict = {}
for img in iname_sorted:
    # Tracked bboxes; shape (m, 4); row (xmin, ymin, width, height)
    bbox_tracks = box_dict[img]
    # detection bboxes; shape (m, 4); row (xmin, ymin, width, height)
    idx_det = idx + 1
    bbox_detec = box_dict[iname_sorted[idx_det]]
    # Assigns detected bboxes to tracked bboxes using IoU as a distance metric
    # returns Tuple containing matches
    # result[0]: matches (track_idx, detection_idx)
    # result[1]: unmatched detections, idx
    # result[2]: unmatched tracks, idx
    result = motrackers.centroid_kf_tracker.assign_tracks2detection_centroid_distances(bbox_tracks, bbox_detec, distance_threshold=10)
    img_names = img+"_"+iname_sorted[idx_det]
    match_dict[img_names] = result
    idx += 1
    # TODO bedingung dass idx < len
    if idx == len(iname_sorted):
        break

for key in match_dict:
    result = match_dict[key]
    pairs = result[0]
    n_pairs = len(pairs)
    print(n_pairs)
    unmatch_d = box_dict[iname_sorted[idx_det]][result[1]]
    unmatch_t = box_dict[img][result[2]]

x=1