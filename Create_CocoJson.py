from PIL import Image
import numpy as np
import json
import os

"""Creates json file in COCO format containing the bounding box characteristics (e.g. coordinates) of annotations or detections
(used for calculation of evaluation metrics)"""

def initialize_COCO_annot():
    """Creates general info for COCO json file"""
    info = {
        'year': '2022',
        'version': [],
        'description': 'Bubble nucleation',
        'contributor': [],
        'url': [],
        'date_created': []
    }
    licenses = [
        {
            'url': [],
            'id': [],
            'name': []
        }
    ]

    categories = [
        {
            'id': 1,
            'name': 'Bubble',
            'supercategory': []
        }
    ]
    images = []
    annotations = []
    return info, licenses, categories, images, annotations

def create_coco_annot():
    """Create a json file containing bounding boxes from annotation"""
    # Create the additional info, save to list 'annotations'
    inf, lic, cat, imgs, annots = initialize_COCO_annot()
    jsondict = {"info":inf,"licenses":lic,"images":imgs,"categories":cat,"annotations":annots}
    return jsondict

def add_annot_to_dict(bbox_list,jsondict,img_id,img_height,img_width,annotation_id):
    """Adds info of current image and all respective annotated bounding boxes [ymin, xmin, ymax, xmax]
    to COCO annotation dictionary"""
    # append image info to dict["images"]
    im_info = {
        "file_name":img_id,
        "height":img_height,
        "width":img_width,
        "date_captured":[],
        "id": img_id
    }
    jsondict["images"].append(im_info)
    # append all bbox annotations to dict["annotations"]
    #annotation_id = 1
    for box in bbox_list:
        annotation = create_bubble_annotation(box, img_id, annotation_id)
        jsondict["annotations"].append(annotation)
        annotation_id += 1
    # return annot_id and take as input to function
    return jsondict, annotation_id

def create_bubble_annotation(bbox_orig, image_id, annotation_id):
    """input: bbox [ymin, xmin, ymax, xmax]
    output format: [xmin, ymin, width, height]"""
    # Find contours (boundary lines)
    # Calculate the coco bounding box and area
    # ORIGINAL x, y, max_x, max_y = bbox_orig
    ymin, xmin, ymax, xmax = bbox_orig
    width = xmax - xmin
    height = ymax - ymin
    bbox = (xmin, ymin, width, height)
    area = width*height
    # annotation
    segmentations = []
    category_id = 1 # only class 'bubble'
    is_crowd = 0
    annotation = {
        'segmentation': segmentations,
        'iscrowd': is_crowd,
        'image_id': image_id,
        'category_id': category_id,
        'id': annotation_id, # unique for each bubble instance
        'bbox': bbox,
        'area': area
    }
    return annotation

def create_coco_results(pred_dict, img_id, jsonlist):
    """Create a json file containing bounding boxes from prediction"""
    bboxes = pred_dict["detection_boxes"]
    box_id = 1
    for box in bboxes:
        # get score of current predicted bbox
        box_score = pred_dict["detection_scores"][box_id-1]
        # get info about current predicted bbox
        box_info = create_bubble_result(box,img_id,box_id,box_score)
        jsonlist.append(box_info)
        box_id += 1
    return jsonlist

def create_bubble_result(bbox_orig, image_id, pred_id, pred_score):
    """input: bbox [ymin, xmin, ymax, xmax]
    output format: [xmin, ymin, width, height]"""
    # Find contours (boundary lines)
    # Calculate the coco bounding box and area
    ymin, xmin, ymax, xmax = bbox_orig
    width = xmax - xmin
    height = ymax - ymin
    bbox = (xmin.astype(float), ymin.astype(float), width.astype(float), height.astype(float))
    area = width*height
    annotation = {
        'image_id': image_id,
        #'id': pred_id,
        'category_id': 1,
        'score': pred_score.astype(float),
        'segmentation': [],
        'iscrowd': 0,
        'bbox': bbox,
        'area': area.astype(float)
    }
    return annotation

def save_json_file(json_content, save_path, name):
    """Saves content of annotation dict or results/prediction list to json file"""
    if "Annot" in name:
        # save annotations as json file
        with open(os.path.join(save_path,(name+'_Annot.json')), 'w') as f:
            #indent for readability (4 spaces on each indent)
            json.dump(json_content, f, indent=4)
            print("New json file: ",name)
    else:
        # save results as json file
        with open(os.path.join(save_path,(name+'.json')), 'w') as f:
            #indent for readability (4 spaces on each indent)
            json.dump(json_content, f, indent=4)
            print("New json file: ",name)
