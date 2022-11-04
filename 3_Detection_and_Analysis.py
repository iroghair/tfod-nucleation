# imports
from numpy.lib.function_base import append
from TFODPaths import get_paths_and_files
from Create_CocoJson import add_annot_to_dict, create_coco_annot, create_coco_results, save_json_file
from DetectAnalysis_Functions import get_image, exclude_partial_pred, get_absolute_pixels, get_time_diff_name, get_visualization_colors, get_pred_thresh, plot_avrg_Bdiam, plot_avrg_Bdiam_sqrt, unite_detection_dicts, Bdiams_over_t
from DetectAnalysis_Functions import unite_detection_dicts, plot_Bcount, hist_Bdiameter, hist_compare_Bdiameter, boxplot_compare_Bdiameter, hist_all_pred_diams

import os
import tensorflow as tf
from object_detection.utils import visualization_utils as viz_utils
from object_detection.utils import label_map_util
from object_detection.builders import model_builder
from object_detection.utils import config_util

import cv2
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import xmltodict
import pprint
import csv

"""Make detections on images and analyze bounding box characteristics (e.g. bubble number, average bubble size etc.)"""

# indicate custom model & desired checkpoint from training
CUSTOM_MODEL = 'my_centernet_hg104_1024_8'
CUSTOM_CHECKPOINT = 'ckpt-21'
# max. allowed detectionss
max_detect = 1000

# get paths and files of custom model
paths, files = get_paths_and_files(CUSTOM_MODEL)

##### LOAD TRAIN MODEL FROM CHECKPOINT #####
# Load pipeline config and build a detection model
configs = config_util.get_configs_from_pipeline_file(files['PIPELINE_CONFIG'])
configs["model"].center_net.object_center_params.max_box_predictions = max_detect
configs['train_config'].max_number_of_boxes = max_detect

detection_model = model_builder.build(model_config=configs['model'], is_training=False)

# Restore checkpoint
ckpt = tf.compat.v2.train.Checkpoint(model=detection_model)
ckpt.restore(os.path.join(paths['CHECKPOINT_PATH'], CUSTOM_CHECKPOINT)).expect_partial()

# set detection function
@tf.function
def detect_fn(image):
    image, shapes = detection_model.preprocess(image)
    prediction_dict = detection_model.predict(image, shapes)
    detections = detection_model.postprocess(prediction_dict, shapes)
    return detections

# classes from custom model (eg {"id": 0, "name": "Bubble"})
category_index = label_map_util.create_category_index_from_labelmap(files['LABELMAP'])

def generate_detections(img):
    """generate detections for test image:
    contains i.a. bounding boxes with [ymin, xmin, ymax, xmax]
    output: dict containing i.a.
            - detection_boxes
            - detection_scores
            - detection_classes ..."""
    #img = cv2.imread(img_path)
    image_np = np.array(img)
    # converting img to tensor
    input_tensor = tf.convert_to_tensor(np.expand_dims(image_np, 0), dtype=tf.float32)
    detect_dict = detect_fn(input_tensor)
    num_detections = int(detect_dict.pop('num_detections'))
    detect_dict = {key: value[0, :num_detections].numpy()
                for key, value in detect_dict.items()}
    detect_dict['num_detections'] = num_detections
    # detection_classes should be ints.
    detect_dict['detection_classes'] = detect_dict['detection_classes'].astype(np.int64)
    return detect_dict

def get_pred_bubble_diameter(img, bounding_boxes, normalized_coord=True):
    """Get diameter of detected bubbles (averaged bounding box width)
    output: df of bubble diameters in mm
    USE ONLY IF NORMALIZED COORDINATES WERE GENERATED"""
    #img = cv2.imread(img_path)
    im_height, im_width, channels = img.shape
    diameter_list = []
    for box in bounding_boxes:
        ymin, xmin, ymax, xmax = box[0], box[1], box[2], box[3]
        # absolute coordinates of box
        if normalized_coord:
            (left, right, top, bottom) = (xmin * im_width, xmax * im_width,
                                        ymin * im_height, ymax * im_height)
        else:
            (left, right, top, bottom) = (xmin, xmax, ymin, ymax)
        b_width = right - left
        b_height = bottom - top
        # convert from pixel to mm
        factor_width = img_width_mm / im_width
        factor_height = img_width_mm / im_width
        b_width_mm = factor_width * b_width
        b_height_mm = factor_height * b_height
        # average width and height
        avg_diameter = (b_width_mm+b_height_mm)/2
        diameter_list.append(avg_diameter)
        #diameter_df = pd.DataFrame(diameter_list)
    return diameter_list

def get_annotated_Diameters(dict):
    # input: annot dict of one img
    im_width = int(dict["annotation"]["size"]["width"])
    im_height = int(dict["annotation"]["size"]["height"])
    diameter_list = []
    # list of object annotations (bubbles)
    dict_list = dict["annotation"]["object"]
    boxes = []
    # iterate through all bubble objects
    for obj in dict_list:
        box = obj["bndbox"]
        # [absolute values]
        ymin, xmin, ymax, xmax = int(box['ymin']), int(box['xmin']), int(box['ymax']), int(box['xmax'])
        b_width = xmax - xmin
        b_height = ymax - ymin
        # convert from pixel to mm
        factor_width = img_width_mm / im_width
        factor_height = img_width_mm / im_width
        b_width_mm = factor_width * b_width
        b_height_mm = factor_height * b_height
        # average width and height
        avg_diameter = (b_width_mm+b_height_mm)/2
        #avg_diameter = b_width_mm
        # save all bubble diameters of img in list
        diameter_list.append(avg_diameter)
        # save detection boxes (annot)
        #boxes.append([ymin,xmin,ymax,xmax])
        # save detection boxes [normalized values] (annot)
        boxes.append([ymin/im_height,xmin/im_width,ymax/im_height,xmax/im_width])
    # dict containing annotated boxes (class 0, score 1 for every box)
    box_dict = {}
    box_dict['detection_boxes'] = np.array(boxes)
    box_dict['detection_classes'] = np.zeros(len(dict_list), dtype=int)
    box_dict['detection_scores'] = np.ones(len(dict_list), dtype=int)
    return diameter_list, box_dict

def visualize_detections(img,detect_dict,score_thresh,save_name,color_id,
                        norm_coord=True,discard_labels=True,discard_scores=True,discard_track_ids=True):
    """Visualize generated detections in test image"""
    #img = cv2.imread(img_path)
    image_np = np.array(img)
    label_id_offset = 1
    image_np_with_detections = image_np.copy()
    # visualization
    detect_vis = viz_utils.visualize_boxes_and_labels_on_image_array(
                image_np_with_detections,
                # detection boxes
                detect_dict['detection_boxes'],
                # name of the class
                detect_dict['detection_classes']+label_id_offset,
                # detection scores
                detect_dict['detection_scores'],
                category_index,
                # color
                track_ids=color_id,
                # specify in absolute (pixel) or normalized coordinates
                # before set to true
                use_normalized_coordinates=norm_coord,
                line_thickness=15, # default 6; new imgs 15; imgs Allessandro 8
                # max. number of boxes to draw
                max_boxes_to_draw=max_detect,
                min_score_thresh=score_thresh,
                # Set the display of notes next to box
                skip_labels=discard_labels,
                skip_scores=discard_scores,
                skip_track_ids=discard_track_ids)
    tested_fig = plt.imshow(cv2.cvtColor(image_np_with_detections, cv2.COLOR_BGR2RGB))
    # save visualization in "tested" directory
    save_path = os.path.join(model_tested_path,save_name)
    plt.savefig(save_path)
    plt.close()
    print("Visualized Detections saved in: ", str(save_path))

#################################
# START PREDICTION ON TEST IMAGES
#################################

# set path of test images
#test_path=os.path.join(paths['IMAGE_PATH'], 'test')
test_path=os.path.join(paths['IMAGE_PATH'], '221103')
# test_path=os.path.join(paths['IMAGE_PATH'], 'Exp2_Tracking')
#test_path=os.path.join(paths['IMAGE_PATH'], 'Distr_Exp4')

MIN_SCORE_THRESH = 0.5

# indicate width and height of imgs [mm]
img_width_mm = 20 #20 #17
img_height_mm = 20 #20 #17

# rescaling image
# ONLY FOR IMAGES WITHOUT ANNOTATIONS (with annot: 1)
scaling_factor = 1

##################################
TESTIMGS_PATHS = []
TESTANNOT_PATHS = []
# get paths of images that should get tested + annotation paths
for file in os.listdir(test_path):
    if not file.startswith('bg'):
        # png format from artificial; tif from actual bubbe imgs
        if file.endswith(".png") or file.endswith(".tif") or file.endswith(".jpg"):
            img_path = os.path.join(test_path, file)
            TESTIMGS_PATHS.append(img_path)
        # xml format for annotation files
        elif file.endswith(".xml"):
            annot_path = os.path.join(test_path, file)
            TESTANNOT_PATHS.append(annot_path)
print("Test Images:\n","\n".join(TESTIMGS_PATHS))
print("Annotations:\n","\n".join(TESTANNOT_PATHS))
# get file extensions of images
img_format = os.path.splitext(TESTIMGS_PATHS[0])[1]
# get background image
back_img = cv2.imread(os.path.join(test_path,('bg_img'+img_format)))

# get last part of test image path (folder name)
folder_name = os.path.basename(os.path.normpath(test_path))
# make path to store tested images to (imgs with bounding boxes)
model_tested_path = os.path.join(paths['IMAGE_PATH'],'tested',CUSTOM_MODEL,(folder_name+'_thresh'+str(MIN_SCORE_THRESH)))
if not os.path.exists(model_tested_path):
    if os.name == 'posix':
        os.makedirs(model_tested_path)
    if os.name == 'nt':
        os.makedirs(model_tested_path)

# Open annotation xml files and save contents to dict
annots_dict = {}
for path in TESTANNOT_PATHS:
    annot_name = os.path.basename(os.path.normpath(path))
    with open(path, 'r', encoding='utf-8') as file:
        my_xml = file.read()
        annots_dict[annot_name] = xmltodict.parse(my_xml)

########### ANNOTATION ############
annot_detect = {}
annot_detect_thresh = {}
annot_detect_abs = {}
annot_diameters = {}
stat_annotDiam = {}
# initialize COCO json file
if os.path.isfile(os.path.join(model_tested_path,"COCO_annot.json")):
    CREATE_COCO_Annot = False
    print("Coco Annotation file already existing")
else:
    annot_coco_dict = create_coco_annot()
    CREATE_COCO_Annot = True
color_id_a = {}
annot_id = 1
for apath in TESTANNOT_PATHS:
    annot_name = os.path.basename(os.path.normpath(apath))
    # Save annotated bubble diameters to dict
    annot_diameters[annot_name], annot_detect[annot_name] = get_annotated_Diameters(annots_dict[annot_name])
    # remove file extension
    iname = annot_name.split('.xml')[0]
    ipath = os.path.join(test_path,iname+img_format)
    im = get_image(ipath)
    # Set color for visualization of boxes (98=red, 118=dark red, 115)
    color_id_a[iname] = np.full(len(annot_detect[annot_name]["detection_boxes"]), 98, dtype=int)
    # visualize annotated bubbles
    visualize_detections(im,annot_detect[annot_name],MIN_SCORE_THRESH,(iname+'_Annot'+img_format),color_id_a[iname])
    # Statistical summary of data
    stat_annotDiam[annot_name] = pd.DataFrame(annot_diameters[annot_name]).describe()
    # thresholded
    annot_detect_thresh[annot_name] = get_pred_thresh(annot_detect[annot_name],MIN_SCORE_THRESH,pred=False)
    if CREATE_COCO_Annot:
        # Add image info and current bounding boxes to COCO dict
        im_height=int(annots_dict[annot_name]["annotation"]["size"]["height"])
        im_width=int(annots_dict[annot_name]["annotation"]["size"]["width"])
        # get absolute pixel values [ymin, xmin, ymax, xmax]
        annot_detect_abs[annot_name]=get_absolute_pixels(annot_detect_thresh[annot_name],im)
        # only take n annotated boxes into account (no boxes with score=0)
        # COCO format bbox: [x, y, width, height]
        annot_coco_dict,annot_id = add_annot_to_dict(annot_detect_abs[annot_name]["detection_boxes"],annot_coco_dict, iname, im_height, im_width,annot_id)
if CREATE_COCO_Annot:
    save_json_file(annot_coco_dict,model_tested_path,"COCO_annot")

############ PREDICTION #############
color_id_p = {}
stat_BubbleDiam = {}
detect = {}
detect_wo_partials = {}
detect_wop_thresh = {}
detect_wop_thresh_abs = {}
bubble_diameters = {}
bubble_number = {}
avrg_diam = {}
# COCO results json file: test if exists
if os.path.isfile(os.path.join(model_tested_path,"COCO_results.json")):
    CREATE_COCO_result = False
    print("Coco Results file already existing")
else:
    resultslist = []
    CREATE_COCO_result = True

# image naming
if folder_name != 'test':
    TESTIMGS_PATHS.sort()
    name_test = os.path.basename(os.path.normpath(TESTIMGS_PATHS[0]))
    if ("t" not in name_test) and ("image" not in name_test):
        imgnames_t_diff = get_time_diff_name(TESTIMGS_PATHS)
i = 0 # idx for img naming

for ipath in TESTIMGS_PATHS:
    # get img name (last part of img_path)
    img_name = os.path.basename(os.path.normpath(ipath))
    iname = img_name.split(img_format)[0]
    im = get_image(ipath, scaling_factor)
    # save detection to dict [ymin, xmin, ymax, xmax]
    ipred = generate_detections(im)
    # check if Allessandros imgs (containing "t" in name)
    if ("t" not in img_name) and ("image" not in img_name):
        img_name = imgnames_t_diff[i]
    detect[img_name] = ipred
    # exclude detections that are cut off
    detect_wo_partials[img_name] = exclude_partial_pred(ipred,im,abs_dist=2)
    # save bounding boxes with min threshold to dict
    detect_wop_thresh[img_name] = get_pred_thresh(detect_wo_partials[img_name],MIN_SCORE_THRESH,pred=True)
    # Set color for visualization of boxes (102 = green)
    color_id_p[iname] = get_visualization_colors(detect_wo_partials[img_name],scaling_factor,MIN_SCORE_THRESH,ipath)
    # visualize thresholded detections (saved in tested directory), show scores
    visualize_detections(im,detect_wo_partials[img_name],MIN_SCORE_THRESH,iname,color_id_p[iname],discard_scores=False)
    # save thresholded bubble diameters to dict
    bubble_diameters[img_name] = get_pred_bubble_diameter(im,detect_wop_thresh[img_name]["detection_boxes"])
    # statistical summary of diameters
    if bubble_diameters[img_name]:
        stat_BubbleDiam[img_name] = pd.DataFrame(bubble_diameters[img_name]).describe()
        # get average bubble diameter (modified detection)
        avrg_diam[img_name] = sum(bubble_diameters[img_name])/len(bubble_diameters[img_name])
    # get number of detected bubbles (modified detection)
    bubble_number[img_name] = len(bubble_diameters[img_name])
    if CREATE_COCO_result:
        # get absolute pixel values
        detect_wop_thresh_abs[img_name]=get_absolute_pixels(detect_wop_thresh[img_name],im)
        # add bboxes (all scores) to COCO results list
        # COCO format bbox: [x, y, width, height]
        resultslist = create_coco_results(detect_wop_thresh_abs[img_name],iname,resultslist)
    i += 1 # raise counter for img naming
# save COCO results to json file
if CREATE_COCO_result:
    save_json_file(resultslist,model_tested_path,"COCO_results")

######### SAVE DIAMETERS ###########
np.save("Db_diaphragm.npy", bubble_diameters)
np.save("Db_mean_diaphragm.npy", avrg_diam)
np.save("Nb_diaphragm.npy", bubble_number)

########### PLOTS ############
ap_detect = {}
color_id_ap = {}
if TESTANNOT_PATHS:
    for ipath in TESTIMGS_PATHS:
        im = get_image(ipath, scaling_factor)
        img_name = os.path.basename(os.path.normpath(ipath))
        # remove file extension from image name
        iname = os.path.splitext(img_name)[0]
        # Diameters
        d_annot=annot_diameters[(iname+'.xml')]
        d_pred=bubble_diameters[img_name] # thresholded and wo partials
        # set histogram bins (same range for all hists)
        bmin = min([min(d_pred),min(d_annot)])
        bmax = max([max(d_pred),max(d_annot)])
        d_bins = np.linspace(bmin,bmax,25)
        # Bubble Diam Hist of Prediction
        diameter_hist_pred = hist_Bdiameter(d_pred,d_bins,("Pred_"+iname), model_tested_path)
        # Bubble Diam Hist of Annotation
        diameter_hist_annot = hist_Bdiameter(d_annot,d_bins,("Annot_"+iname), model_tested_path)
        # Comparison from Annotation and Prediction (Histogram)
        diameter_hist_comp = hist_compare_Bdiameter(d_pred,d_annot,d_bins,iname,model_tested_path)
        # Comparison from Annotation and Prediction (Boxplot)
        diameter_boxpl_comp = boxplot_compare_Bdiameter(d_pred,d_annot,iname,model_tested_path)
        # visualize pred boxes in preexisting annot box visualization (thresholded and wo partials)
        color_id_ap[iname] = np.concatenate((color_id_p[iname],color_id_a[iname]),axis=0)
        ap_detect[iname] = unite_detection_dicts(detect_wo_partials[img_name],annot_detect_thresh[(iname+'.xml')],MIN_SCORE_THRESH)
        visualize_detections(im,ap_detect[iname],MIN_SCORE_THRESH,
                            (iname+'_CompViz'+img_format),color_id_ap[iname],norm_coord=False)
else:
    bmins = []
    bmaxs = []
    # only plot prediction histogram (no comparative plots)
    for j in bubble_diameters.keys():
        if j !="0 s":
            if bubble_diameters[j]:
                # set histogram bins (same range for all hists)
                bmin = min(bubble_diameters[j])
                bmax = max(bubble_diameters[j])
                bmins.append(bmin)
                bmaxs.append(bmax)
                d_bins = np.linspace(bmin,bmax,25)
                # Bubble Diam from Prediction
                diameter_hist_pred = hist_Bdiameter((bubble_diameters[j]),d_bins,("Pred_"+j),model_tested_path)
                # Bubble Diam from Prediction
                diameter_hist_pred = hist_Bdiameter((bubble_diameters[j]),d_bins,("Pred_"+j),model_tested_path)

# do not create time series plots for artificial imgs   
if folder_name != 'test':
    # predicted average bubble diam. + solution from ODE
    plot_avrg_Bdiam(avrg_diam,test_path,model_tested_path)
    plot_avrg_Bdiam_sqrt(avrg_diam,test_path,model_tested_path)
    # all diameters over time
    Bdiams_over_t(bubble_diameters,model_tested_path)
    # str split for image names in function needs to be adjusted before generating plot!
    plot_Bcount(bubble_number,test_path,model_tested_path)
    # joint histograms of prediction
    # check arrangement of subplots (depending on number of images)!
    bins = np.linspace(min(bmins),max(bmaxs),25)
    #bubble_diameters_copy = dict(bubble_diameters)
    #bubble_diameters_copy.pop("0")
    hist_all_pred_diams(bubble_diameters,bins,test_path,model_tested_path)

# save into textfiles
l=[]
[l.append([k,v]) for k,v in avrg_diam.items()]
textfile1=open("avrg_diam_Exp2Tracking.txt","w")
for element in l:
    textfile1.write(str(element) + "\n")
textfile1.close()

u=[]
[u.append([k,v]) for k,v in bubble_number.items()]
textfile2=open("bubble_number_Exp2Tracking.txt","w")
for element in u:
    textfile2.write(str(element) + "\n")
textfile2.close()