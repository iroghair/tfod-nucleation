from TFODPaths import get_paths_and_files

import os
import tensorflow as tf
from object_detection.utils import visualization_utils as viz_utils
from object_detection.utils import label_map_util
from object_detection.builders import model_builder
from object_detection.utils import config_util

import cv2 
import numpy as np
from matplotlib import pyplot as plt

CUSTOM_MODEL = 'my_faster_rcnn_resnet50_4'
CUSTOM_CHECKPOINT = 'ckpt-21'

# get paths and files of custom model
paths, files = get_paths_and_files(CUSTOM_MODEL)

##### LOAD TRAIN MODEL FROM CHECKPOINT #####
# Load pipeline config and build a detection model
configs = config_util.get_configs_from_pipeline_file(files['PIPELINE_CONFIG'])
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

##### DETECT FROM AN IMAGE #####
def generate_detections(img_path):
    """generate detections for test image:
    contains i.a. bounding boxes with [ymin, xmin, ymax, xmax]"""
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

def get_pred_bubble_diameter(img_path, detect_dict):
    """Get diameter of detected bubble (averaged bounding box width)
    USE ONLY IF NORMALIZED COORDINATES WERE GENERATED"""
    img = cv2.imread(img_path)
    bounding_boxes = detect_dict['detection_boxes']
    im_height, im_width, channels = img.shape
    diameter_list = []
    for box in bounding_boxes:
        ymin, xmin, ymax, xmax = box[0], box[1], box[2], box[3]
        # absolute coordinates of box
        (left, right, top, bottom) = (xmin * im_width, xmax * im_width,
                                    ymin * im_height, ymax * im_height)
        b_width = right - left
        b_height = top - bottom
        avg_diameter = (b_width+b_height)/2
        diameter_list.append(avg_diameter)
    return diameter_list

def get_bubble_pred_thresh(detect_dict,score_thresh):
    """Get all detections with score over threshold score_thresh"""
    scores = detect_dict['detection_scores']
    classes = detect_dict['detection_classes']
    boxes = detect_dict['detection_boxes']
    #print((i,scores[i]) for i in boxes if scores[i] > score_thresh)
    #print((i,scores[i]) for i,v in enumerate(boxes) if scores[i] > score_thresh)
    pred_thresh = []
    for i, v in enumerate(boxes):
        if scores[i] >= score_thresh:
            print(v)
            pred_thresh.append(v)
    return pred_thresh

def visualize_detections(img_path, detect_dict, score_thresh):
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
                # specify in absolute (pixel) or normalized coordinates
                use_normalized_coordinates=True,
                # TODO set max. count
                max_boxes_to_draw=100,
                min_score_thresh=score_thresh,
                agnostic_mode=False)
    tested_fig = plt.imshow(cv2.cvtColor(image_np_with_detections, cv2.COLOR_BGR2RGB))
    # save visualization in "tested" directory
    save_path = os.path.join(model_tested_path,img_name)
    plt.savefig(save_path)


#################################
# START PREDICTION ON TEST IMAGES

TESTIMGS_PATHS = []
bubble_diameters = {}
bubble_detections = {}
bubble_detections_thresh = {}
MIN_SCORE_THRESH = 0.5

category_index = label_map_util.create_category_index_from_labelmap(files['LABELMAP'])

# path to store tested images to (imgs with bounding boxes)
model_tested_path = os.path.join(paths['IMAGE_PATH'],'tested',CUSTOM_MODEL)
if not os.path.exists(model_tested_path):
    if os.name == 'posix':
        os.mkdir(model_tested_path)
    if os.name == 'nt':
        os.mkdir(model_tested_path)

# get paths of all images that should get tested
# (list all entries of test img directory)
for file in os.listdir(os.path.join(paths['IMAGE_PATH'], 'test')):
    if file.endswith(".png"):
        img_path = os.path.join(paths['IMAGE_PATH'], 'test', file)
        TESTIMGS_PATHS.append(img_path)
print(TESTIMGS_PATHS)


for ipath in TESTIMGS_PATHS:
    # get img name (last part of img_path)
    img_name = os.path.basename(os.path.normpath(ipath))
    # save detection to dict
    bubble_detections[img_name] = generate_detections(ipath)
    ipred = bubble_detections[img_name]
    # save bubble diameters to dict
    bubble_diameters[img_name] = get_pred_bubble_diameter(ipath, ipred)
    # save detections with min threshold to dict
    bubble_detections_thresh[img_name] = get_bubble_pred_thresh(ipred,MIN_SCORE_THRESH)
    # visualize detections
    visualize_detections(ipath,ipred,MIN_SCORE_THRESH)

x = 1