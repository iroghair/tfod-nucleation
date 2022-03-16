from TFODPaths import get_paths_and_files
from SeparateTest_Functions import get_image
import os
import json
import cv2 
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
import motrackers.centroid_kf_tracker
from object_detection.utils import visualization_utils as viz_utils

"""Load json file of ground truth and detection"""
# indicate custom model & desired checkpoint from training
CUSTOM_MODEL = 'my_centernet_hg104_1024_7'
TESTED_IMG_FOLDER = 'supersaturation_0.16_thresh0.5'
IMG_FOLDER = 'supersaturation_0.16'
img_format = '.tif'
tracker_threshold = 50
img_width_mm = 20

def get_bdiams(img,bounding_boxes):
    """Get diameter of bubbles (averaged bounding box width, height)
    output: df of bubble diameters in mm"""
    im_height, im_width, channels = img.shape
    diameter_list = []
    for box in bounding_boxes:
        xmin, ymin, width, height = box[0], box[1], box[2], box[3]
        # convert from pixel to mm
        factor_width = img_width_mm / im_width
        factor_height = img_width_mm / im_width
        b_width_mm = factor_width * width
        b_height_mm = factor_height * height
        # average width and height
        avg_diameter = (b_width_mm+b_height_mm)/2
        diameter_list.append(avg_diameter)
        #diameter_df = pd.DataFrame(diameter_list)
    return diameter_list

def visualize_unmatched_bboxes(unmatched_t,iname,save_name,
                        norm_coord=True,discard_labels=True,discard_scores=True,discard_track_ids=True):
    """Visualize matches bounding boxes
    input: tuple containing unmatched bboxes in track img"""
    track_img_path = os.path.join(paths['IMAGE_PATH'],IMG_FOLDER,iname+img_format)
    img_track = get_image(track_img_path,bg_img=None)
    track_i_np = np.array(img_track)
    track_i_np_bb = track_i_np.copy()
    # Change bbox format: [xmin, ymin, width, height] --> [ymin, xmin, ymax, xmax]
    bboxes = []
    for box in unmatched_t:
        xmin, ymin, width, height = box[0], box[1], box[2], box[3]
        ymax = ymin + height
        xmax = xmin + width
        bboxes.append([ymin,xmin,ymax,xmax])
    bboxes = np.asarray(bboxes)
    # visualization
    detect_vis = viz_utils.visualize_boxes_and_labels_on_image_array(
                track_i_np_bb,
                # detection boxes
                bboxes,
                # name of the class
                np.ones(len(boxes)),
                # detection scores
                np.ones(len(boxes)),
                # category index
                {"id": 0, "name": "Bubble"},
                # colors
                track_ids=np.full(len(boxes), 98, dtype=int),
                line_thickness=6, # default 6
                max_boxes_to_draw=n_unmatch_max, # default 20
                skip_labels=discard_labels,
                skip_scores=discard_scores,
                skip_track_ids=discard_track_ids)
    tested_fig = plt.imshow(cv2.cvtColor(track_i_np_bb, cv2.COLOR_BGR2RGB))
    # save visualization in "tested" directory
    save_path = os.path.join(model_tested_path,(iname+save_name))
    plt.savefig(save_path)
    plt.close()
    print("Visualization of unmatched bboxes in: ", str(save_path))

def hist_all_pred_diams(dict,hist_bins,save_path):
    """HISTOGRAMS OF DETACHMENT BUBBLE DIAMETER FROM ALL IMAGES (one subaxis per image)
    input: unmatched bubble diameters"""
    new_keys=[]
    old_keys = list(dict)
    # change keys of dict to respective time stamp
    for k in old_keys:
        # remove tail of key string
        splits = k.split('_')[0]
        # remove "t" in front of time indication
        times_str = splits.split('t')[1]
        new_keys.append(times_str)
        # replace old key with new key
        dict[times_str] = dict.pop(k)
    # sort keys by timestamp
    new_keys.sort()
    # intialize figure
    fig, ((ax1,ax2,ax3), (ax4,ax5,ax6), (ax7,ax8,ax9)) = plt.subplots(3, 3, sharex=True, sharey=True)
    fig.suptitle(IMG_FOLDER)
    ax_names = [ax1,ax2,ax3,ax4,ax5,ax6,ax7,ax8,ax9]
    plt.rcParams.update({'axes.titlesize':'small'})
    plt.rcParams.update({'axes.titlepad':1}) 
    # plot dict entries in individual histograms
    for i,nk in enumerate(new_keys):
        sns.histplot(dict[nk],ax=ax_names[i],bins=hist_bins,kde=True,color='darkblue',stat='density')
        ax_names[i].set_title(("t = "+nk+" min"))
        ax_names[i].set(ylabel=None)
    fig.text(0.5, 0.04, 'Detachment diameter [mm]', ha='center', va='center')
    fig.text(0.06, 0.5, 'Probability density', ha='center', va='center', rotation='vertical')
    # spacing between subplots
    plt.subplots_adjust(wspace=0.1,hspace=0.2)
    plt.savefig(os.path.join(save_path,f'DetachDiam_Hist.png'))
    plt.close()
    print(f'Detachment Diameter Hist saved under {save_path}.')

### START ###########################
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

# bbox format [x, y, width, height]
box_dict = {}
for key in dict:
    d = dict[key]
    boxes = pd.DataFrame(d['bbox'].to_list(),columns=['x','y','width','height'])
    box_dict[key] = boxes.to_numpy()

iname_sorted = sorted(box_dict.keys())
idx = 0
match_dict = {}
unmatch_bb_d = {}
unmatch_bb_t = {}
for iname in iname_sorted:
    # Tracked bboxes; shape (m, 4); row (xmin, ymin, width, height)
    bbox_tracks = box_dict[iname]
    # detection bboxes; shape (m, 4); row (xmin, ymin, width, height)
    bbox_detec = box_dict[iname_sorted[idx+1]]
    # Assigns detected bboxes (n+1) to tracked bboxes (n) using IoU as a distance metric
    # returns Tuple containing matches
    # result[0]: matches (track_idx, detection_idx)
    # result[1]: unmatched detections, idx
    # result[2]: unmatched tracks, idx
    result = motrackers.centroid_kf_tracker.assign_tracks2detection_centroid_distances(bbox_tracks, bbox_detec, distance_threshold=tracker_threshold)
    img_names = iname+"_"+iname_sorted[idx+1]
    match_dict[img_names] = result
    unmatch_bb_d[iname_sorted[idx+1]] = box_dict[iname_sorted[idx+1]][result[1]]
    unmatch_bb_t[iname] = box_dict[iname][result[2]]
    idx += 1
    if idx == len(iname_sorted)-1:
        break

# number of matched bubbles
n_pairs = []
for key in match_dict:
    result = match_dict[key]
    pairs = result[0]
    n_pairs.append(len(pairs))
    print(f'{len(pairs)} matched boxes for {key}')

# number of unmatched bubbles
n_unmatch = []
for key in unmatch_bb_t:
    n_unmatch.append(len(unmatch_bb_t[key]))
    print(f'{len(unmatch_bb_t[key])} unmatched boxes for {key}')
n_unmatch_max = max(n_unmatch)

# sizes of unmatched bubbles [mm]
unmatch_diams_t = {}
bmins = []
bmaxs = []
for iname in unmatch_bb_t:
    track_img_path = os.path.join(paths['IMAGE_PATH'],IMG_FOLDER,iname+img_format)
    img_track = get_image(track_img_path,bg_img=None)
    unmatch_diams_t[iname] = get_bdiams(img_track,unmatch_bb_t[iname])
    bmins.append(min(unmatch_diams_t[iname]))
    bmaxs.append(max(unmatch_diams_t[iname]))

bins = np.linspace(min(bmins),max(bmaxs),25)
hist_all_pred_diams(unmatch_diams_t,bins,model_tested_path)

# VISUALIZATION
idx = 0
for iname in iname_sorted:
    # track img
    visualize_unmatched_bboxes(unmatch_bb_t[iname],iname,'_Unmatched')
    # vis. unmatched boxes in following image
    next_iname = iname_sorted[idx+1]
    visualize_unmatched_bboxes(unmatch_bb_t[iname],next_iname,'_LostBboxes')
    #img1 = cv2.imread(os.path.join(model_tested_path,(iname+'_Unmatched')))
    #img2 = cv2.imread(os.path.join(model_tested_path,(next_iname+'_LostBoxes')))
    #vis = cv2.hconcat([img1,img2])
    idx += 1
    if idx == len(iname_sorted)-1:
        break