import cv2 
import numpy as np

def exclude_partial_pred(detect_dict,img_path,abs_dist):
    """Removes bounding boxes from detection dict that touch the image boundaries
    Goal: remove partial detections
    abs_dist = allowed distance [in pixels] of bbox to border of img"""
    img = cv2.imread(img_path)
    im_height, im_width, channels = img.shape
    xdist_rel = abs_dist/im_width
    ydist_rel = abs_dist/im_height
    boxes = detect_dict['detection_boxes']
    scores = detect_dict['detection_scores']
    multscores = detect_dict['detection_multiclass_scores']
    classes = detect_dict['detection_classes']
    strided = detect_dict['detection_boxes_strided']
    rows = [] # rows that will be deleted
    for i, box in enumerate(boxes):
        ymin, xmin, ymax, xmax = box[0], box[1], box[2], box[3]
        # exclude bboxes that touch the image borders
        if (xmin<xdist_rel or ymin<ydist_rel or xmax>(1-xdist_rel) or ymax>(1-ydist_rel)):
            # delete respective rows from arrays in dict
            rows.append(i)
            detect_dict['num_detections'] -= 1
        else:
            pass
    detect_dict['detection_boxes'] = np.delete(boxes, rows, 0)
    detect_dict['detection_scores'] = np.delete(scores, rows, 0)
    detect_dict['detection_multiclass_scores'] = np.delete(multscores, rows, 0)
    detect_dict['detection_classes'] = np.delete(classes, rows, 0)
    detect_dict['detection_boxes_strided'] = np.delete(strided, rows, 0)
    return detect_dict

def get_absolute_pixels(detect_dict,img_path):
    """Gets absolute pixel values for relative bounding boxes"""
    img = cv2.imread(img_path)
    im_height, im_width, channels = img.shape
    boxes = detect_dict['detection_boxes']
    abs_boxes = []
    for box in boxes:
        ymin, xmin, ymax, xmax = box[0], box[1], box[2], box[3]
        # absolute coordinates of box
        (top, left, bottom, right) = (ymin * im_height, xmin * im_width,
                                      ymax * im_height, xmax * im_width)
        abs_boxes.append((top, left, bottom, right))
    detect_dict['detection_boxes'] = np.array(abs_boxes)
    return detect_dict

def get_bubble_pred_thresh(detect_dict,score_thresh):
    """Get bounding boxes with score over threshold score_thresh
    output: numpy array with locations of bounding boxes"""
    scores = detect_dict['detection_scores']
    boxes = detect_dict['detection_boxes']
    #print((i,scores[i]) for i in boxes if scores[i] > score_thresh)
    #print((i,scores[i]) for i,v in enumerate(boxes) if scores[i] > score_thresh)
    pred_thresh = []
    for i, v in enumerate(boxes):
        if scores[i] >= score_thresh:
            pred_thresh.append(v)
    pred_thresh = np.array(pred_thresh)
    return pred_thresh

def get_visualization_colors(detect_dict,color_dict,img_name,area_thresh):
    """Sets colors for bounding boxes"""
    # 102 green
    color_dict[img_name] = np.full(len(detect_dict["detection_boxes"]), 102, dtype=int)
    i_smallest = []
    for i, box in enumerate(detect_dict["detection_boxes"]):
        ymin, xmin, ymax, xmax = box[0], box[1], box[2], box[3]
        bwidth = xmax - xmin
        bheight = ymax - ymin
        area = bwidth*bheight
        if area < area_thresh:
            i_smallest.append(i)
        else:
            pass
    # give the smallest bounding boxes a different color (128 yellow)
    color_dict[img_name][i_smallest] = 128
    return color_dict

