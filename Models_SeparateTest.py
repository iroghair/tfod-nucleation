from pickle import FALSE, TRUE
from numpy.lib.function_base import append
from TFODPaths import get_paths_and_files
from Create_CocoJson import add_annot_to_dict, create_coco_annot, create_coco_results, save_json_file
from SeparateTest_Functions import exclude_partial_pred, get_absolute_pixels, get_visualization_colors, get_bubble_pred_thresh

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

"""Annotations and Checkpoints of Model required as input"""

# indicate custom model & desired checkpoint from training
CUSTOM_MODEL = 'my_centernet_hg104_1024_5'
CUSTOM_CHECKPOINT = 'ckpt-21'
# max. allowed detections
max_detect = 500

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

def generate_detections(img_path):
    """generate detections for test image:
    contains i.a. bounding boxes with [ymin, xmin, ymax, xmax]
    output: dict containing i.a.
            - detection_boxes
            - detection_scores
            - detection_classes ..."""
    img = cv2.imread(img_path)
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

def get_pred_bubble_diameter(img_path, bounding_boxes):
    """Get diameter of detected bubbles (averaged bounding box width)
    output: df of bubble diameters in mm
    USE ONLY IF NORMALIZED COORDINATES WERE GENERATED"""
    img = cv2.imread(img_path)
    im_height, im_width, channels = img.shape
    diameter_list = []
    for box in bounding_boxes:
        ymin, xmin, ymax, xmax = box[0], box[1], box[2], box[3]
        # absolute coordinates of box
        (left, right, top, bottom) = (xmin * im_width, xmax * im_width,
                                    ymin * im_height, ymax * im_height)
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
        #avg_diameter = (b_width_mm+b_height_mm)/2
        avg_diameter = b_width_mm
        # save all bubble diameters of img in list
        diameter_list.append(avg_diameter)
        # save detection boxes [normalized values] (annot)
        boxes.append([ymin/im_height,xmin/im_width,ymax/im_height,xmax/im_width])
    # dict containing annotated boxes (class 0, score 1 for every box)
    box_dict = {}
    box_dict['detection_boxes'] = np.array(boxes)
    box_dict['detection_classes'] = np.zeros(len(dict_list), dtype=int)
    box_dict['detection_scores'] = np.ones(len(dict_list), dtype=int)
    return diameter_list, box_dict

def unite_detection_dicts(dict,img_name,pdict,adict):
    """Unite detection dicts from prediction and annotation"""
    dict[img_name+img_format]={}
    a1=pdict["detection_boxes"]
    a2=adict["detection_boxes"]
    dict[iname+img_format]["detection_boxes"] = np.concatenate((a1, a2), axis=0)
    a3=pdict['detection_classes']
    a4=adict['detection_classes']
    dict[iname+img_format]['detection_classes']=np.concatenate((a3, a4), axis=0)
    a5=pdict['detection_scores']
    a6=adict['detection_scores']
    dict[iname+img_format]['detection_scores']=np.concatenate((a5, a6), axis=0)
    return dict

def visualize_detections(img_path,detect_dict,score_thresh,img_name,color_id,
                        discard_labels=True,discard_scores=True,discard_track_ids=True):
    """Visualize generated detections in test image"""
    img = cv2.imread(img_path)
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
                use_normalized_coordinates=True,
                # max. number of boxes to draw
                max_boxes_to_draw=max_detect,
                min_score_thresh=score_thresh,
                # Set the display of notes next to box
                skip_labels=discard_labels,
                skip_scores=discard_scores,
                skip_track_ids=discard_track_ids)
    tested_fig = plt.imshow(cv2.cvtColor(image_np_with_detections, cv2.COLOR_BGR2RGB))
    # save visualization in "tested" directory
    save_path = os.path.join(model_tested_path,img_name)
    plt.savefig(save_path)
    plt.close()
    print("Visualized Detections saved in: ", str(save_path))

# TODO outsource plots
def hist_Bdiameter(diam,hbins,img_name):
    """Plot histogram and probability density of the number of detected bubbles
        (kernel density estimation - Gaussian)
    input: list containing diameters of all bubbles in one image
           bins in histogram; name of img"""
    sns.histplot(diam,bins=hbins,kde=True,color='darkblue',stat='density')
    plt.xlabel('Bubble diameter [mm]')
    plt.ylabel('Density')
    plt.title(img_name)
    plt.savefig(os.path.join(model_tested_path,f'Diam_HistDens_{img_name}.png'))
    plt.close()
    print(f'Diameter Hist+Dens saved under {model_tested_path} for {img_name}')

def hist_compare_Bdiameter(diam_pred,diam_annot,my_bins,img_name):
    """Plot histogram and probability density of the number of detected bubbles
        (kernel density estimation - Gaussian)
    input: lists containing diameters from prediction and annotation
           bins in histogram; name of img"""
    # plot two histograms in one figure
    sns.histplot(diam_pred,bins=my_bins,kde=True,stat='density',color="green",label="Prediction")
    sns.histplot(diam_annot,bins=my_bins,kde=True,stat='density',color="blue",label="Annotation")
    plt.xlabel('Bubble diameter [mm]')
    plt.ylabel('Density')
    plt.title(img_name)
    plt.legend()
    plt.savefig(os.path.join(model_tested_path,f'Diam_HistComp_{img_name}.png'))
    plt.close()
    print(f'Diameter Compare Hist saved under {model_tested_path} for {img_name}')

def boxplot_compare_Bdiameter(diam_pred,diam_annot,img_name):
    """Boxplots containing statistical summary of diameter data
    input: lists containing diameters from prediction and annotation
           name of image"""
    fig, ax = plt.subplots()
    ax.boxplot([diam_pred,diam_annot])
    ax.set_xticklabels(['Prediction','Annotation'])
    # print number of observations into plot
    n_pred, n_annot = len(diam_pred), len(diam_annot)
    textstr = '\n'.join(["Observations","Predicition: "+str(n_pred),"Annotation: "+str(n_annot)])
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax.text(0.4, 0.95, textstr, transform=ax.transAxes, fontsize=10,verticalalignment='top', bbox=props)
    plt.title('Statistical summary: '+img_name)
    plt.savefig(os.path.join(model_tested_path,f'Diam_Boxplot_{img_name}.png'))
    plt.close()
    print(f'Boxplots saved under {model_tested_path} for {img_name}')

def plot_Bcount(dict):
    """Plot number of detected bubbles over time
    (timestamps must be indicated in image name)
    input: dictionary containing number of detected bubbles
           image names as keys"""
    Bcount = list(dict.values())
    dk=dict.keys()
    # remove tail of key string
    splits = [key.split('_')[0] for key in dk]
    # remove "t" in front of time indication
    times_str = [s.split('t')[1] for s in splits]
    # get timestamps as int
    times = [int(t) for t in times_str]
    plt.plot(times, Bcount, 'o')
    plt.xlabel('Time')
    plt.ylabel('Bubble count [-]')
    name = os.path.basename(os.path.normpath(test_path))
    plt.title("Total number of bubbles: "+name)
    plt.savefig(os.path.join(model_tested_path,f'BCount_{name}.png'))
    plt.close()
    print(f'Bubble count plot saved under {model_tested_path}')

#################################
# START PREDICTION ON TEST IMAGES
#################################

# set path of test images
test_path=os.path.join(paths['IMAGE_PATH'], 'test')
#test_path=os.path.join(paths['IMAGE_PATH'], 'actual_bubbles')
#test_path=os.path.join(paths['IMAGE_PATH'], 'supersaturation_0.16_cropIR')
#test_path=os.path.join(paths['IMAGE_PATH'], 'H2_porousNickel')
#test_path=os.path.join(paths['IMAGE_PATH'], 'output_10_80_JGmask')

MIN_SCORE_THRESH = 0.5

# indicate width and height of imgs [mm]
img_width_mm = 20
img_height_mm = 20

##################################
TESTIMGS_PATHS = []
TESTANNOT_PATHS = []
# get paths of images that should get tested + annotation paths
for file in os.listdir(test_path):
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

# get last part of test image path (folder name)
folder_name = os.path.basename(os.path.normpath(test_path))
# make path to store tested images to (imgs with bounding boxes)
model_tested_path = os.path.join(paths['IMAGE_PATH'],'tested',CUSTOM_MODEL,(folder_name+'_thresh'+str(MIN_SCORE_THRESH)))
if not os.path.exists(model_tested_path):
    if os.name == 'posix':
        os.makedirs(model_tested_path)
    if os.name == 'nt':
        os.makedirs(model_tested_path)

# Open xml files and save contents to dict
annots_dict = {}
for path in TESTANNOT_PATHS:
    annot_name = os.path.basename(os.path.normpath(path))
    with open(path, 'r', encoding='utf-8') as file:
        my_xml = file.read()
        annots_dict[annot_name] = xmltodict.parse(my_xml)

########### ANNOTATION ############
annot_detections = {}
annot_detections_abs = {}
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
for apath in TESTANNOT_PATHS:
    annot_name = os.path.basename(os.path.normpath(apath))
    # Save annotated bubble diameters to dict
    annot_diameters[annot_name], annot_detections[annot_name] = get_annotated_Diameters(annots_dict[annot_name])
    # remove file extension
    iname = annot_name.split('.xml')[0]
    ipath = os.path.join(test_path,iname+img_format)
    # Set color for visualization of boxes (98=red, 118=dark red)
    color_id_a[iname] = np.full(len(annot_detections[annot_name]["detection_boxes"]), 115, dtype=int)
    # visualize annotated bubbles
    #visualize_detections(ipath,annot_detections[annot_name],MIN_SCORE_THRESH,(iname+'_Annot'+img_format),color_id_a[iname])
    # Statistical summary of data
    stat_annotDiam[annot_name] = pd.DataFrame(annot_diameters[annot_name]).describe()
    if CREATE_COCO_Annot:
        # Add image info and current bounding boxes to COCO dict
        im_height=int(annots_dict[annot_name]["annotation"]["size"]["height"])
        im_width=int(annots_dict[annot_name]["annotation"]["size"]["width"])
        # get absolute pixel values
        annot_detections_abs[annot_name]=get_absolute_pixels(annot_detections[annot_name],ipath)
        # only take the 80 annotated boxes into account (no boxes with score=0)
        bboxes = get_bubble_pred_thresh(annot_detections_abs[annot_name],MIN_SCORE_THRESH)
        annot_coco_dict = add_annot_to_dict(bboxes,annot_coco_dict, iname, im_height, im_width)
if CREATE_COCO_Annot:
    save_json_file(annot_coco_dict,model_tested_path,"COCO_annot")

############ PREDICTION #############
color_id_p = {}
stat_BubbleDiam = {}
bubble_detections = {}
ipred_wo_partials = {}
ipred_wo_partials_abs = {}
bubble_diameters = {}
bboxes_thresh = {}
bubble_number = {}
# COCO results json file: test if exists
if os.path.isfile(os.path.join(model_tested_path,"COCO_results.json")):
    CREATE_COCO_result = False
    print("Coco Results file already existing")
else:
    resultslist = []
    CREATE_COCO_result = True
for ipath in TESTIMGS_PATHS:
    # get img name (last part of img_path)
    img_name = os.path.basename(os.path.normpath(ipath))
    iname = img_name.split(img_format)[0]
    # save detection to dict
    ipred = generate_detections(ipath)
    bubble_detections[img_name] = ipred
    # exclude detections that are cut off
    ipred_wo_partials[img_name] = exclude_partial_pred(ipred,ipath,abs_dist=4)
    # save bounding boxes with min threshold to dict
    bboxes_thresh[img_name] = get_bubble_pred_thresh(ipred_wo_partials[img_name],MIN_SCORE_THRESH)
    # Set color for visualization of boxes (102 = green)
    color_id_p = get_visualization_colors(ipred_wo_partials[img_name],color_id_p,iname,0.0005)
    # visualize thresholded detections (saved in tested directory), show scores
    visualize_detections(ipath,ipred_wo_partials[img_name],MIN_SCORE_THRESH,img_name,color_id_p[iname],discard_scores=False)
    # save thresholded bubble diameters to dict
    bubble_diameters[img_name] = get_pred_bubble_diameter(ipath,bboxes_thresh[img_name])
    # statistical summary of diameters
    stat_BubbleDiam[img_name] = pd.DataFrame(bubble_diameters[img_name]).describe()
    # get number of detected bubbles (modified detection)
    bubble_number[img_name] = len(bubble_diameters[img_name])
    if CREATE_COCO_result:
        # get absolute pixel values
        ipred_wo_partials_abs[img_name]=get_absolute_pixels(ipred_wo_partials[img_name],ipath)
        # add bboxes (all scores) to COCO results list
        resultslist = create_coco_results(ipred_wo_partials_abs[img_name],iname,resultslist)
# save COCO results to json file
if CREATE_COCO_result:
    save_json_file(resultslist,model_tested_path,"COCO_results")

########### PLOTS ############
ap_detections = {}
if TESTANNOT_PATHS:
    for ipath in TESTIMGS_PATHS:
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
        #diameter_hist_pred = hist_Bdiameter(d_pred,d_bins,("Pred_"+iname))
        # Bubble Diam Hist of Annotation
        #diameter_hist_annot = hist_Bdiameter(d_annot,d_bins,("Annot_"+iname))
        # Comparison from Annotation and Prediction (Histogram)
        diameter_hist_comp = hist_compare_Bdiameter(d_pred,d_annot,d_bins,iname)
        # Comparison from Annotation and Prediction (Boxplot)
        diameter_boxpl_comp = boxplot_compare_Bdiameter(d_pred,d_annot,iname)
        # visualize pred boxes in preexisting annot box visualization
        # TODO check what goes wrong with coloring the bounding boxes
        color_id_ap = np.concatenate((color_id_p[iname],color_id_a[iname]),axis=0)
        ap_detections[iname] = unite_detection_dicts(ap_detections,iname,ipred_wo_partials[img_name],annot_detections[(iname+'.xml')])
        visualize_detections(ipath,ap_detections[img_name],MIN_SCORE_THRESH,
                            (iname+'_CompViz'+img_format),color_id_ap)
else:
    # only plot prediction histogram (no comparative plots)
    for j in bubble_diameters.keys():
        # set histogram bins (same range for all hists)
        bmin = min(bubble_diameters[j])
        bmax = max(bubble_diameters[j])
        d_bins = np.linspace(bmin,bmax,25)
        # Bubble Diam from Prediction
        diameter_hist_pred = hist_Bdiameter((bubble_diameters[j]),d_bins,("Pred_"+j))
        # Bubble Diam from Prediction
        diameter_hist_pred = hist_Bdiameter((bubble_diameters[j]),d_bins,("Pred_"+j))

# str split needs to be adjusted before generating plot!
plot_Bcount(bubble_number)