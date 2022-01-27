import os
import cv2 
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns

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

def get_visualization_colors(detect_dict,color_dict,img_name,score_thresh,area_thresh):
    """Sets colors for bounding boxes with min. score of score_thresh
    Smallest bounding boxes obtain a different color"""
    scores = detect_dict["detection_scores"]
    boxes = detect_dict["detection_boxes"]
    thresh_boxes = boxes[scores > score_thresh]
    # 102 green
    color_dict[img_name] = np.full(len(thresh_boxes), 102, dtype=int)
    i_smallest = []
    for i, box in enumerate(thresh_boxes):
        ymin, xmin, ymax, xmax = box[0], box[1], box[2], box[3]
        bwidth = xmax - xmin
        bheight = ymax - ymin
        area = bwidth*bheight
        if area.astype(float) < area_thresh:
            i_smallest.append(i)
        else:
            pass
    # give the smallest bounding boxes a different color (128 yellow)
    color_dict[img_name][i_smallest] = 128
    return color_dict

def unite_detection_dicts(dict,img_name,pdict,adict,score_thresh):
    """Unite detection dicts from prediction and annotation
    Predicted detections thresholded my minimum score"""
    dict[img_name]={}
    i_thresh = pdict["detection_scores"] > score_thresh
    a1=pdict["detection_boxes"][i_thresh]
    a2=adict["detection_boxes"]
    dict[img_name]["detection_boxes"] = np.concatenate((a1, a2), axis=0)
    a3=pdict['detection_classes'][i_thresh]
    a4=adict['detection_classes']
    dict[img_name]['detection_classes']=np.concatenate((a3, a4), axis=0)
    a5=pdict['detection_scores'][i_thresh]
    a6=adict['detection_scores']
    dict[img_name]['detection_scores']=np.concatenate((a5, a6), axis=0)
    return dict

############## PLOTS ##########################

def plot_Bcount(dict, test_path, save_path):
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
    plt.savefig(os.path.join(save_path,f'BCount_{name}.png'))
    plt.close()
    print(f'Bubble count plot saved under {save_path}')

def hist_compare_Bdiameter(diam_pred,diam_annot,my_bins,img_name,save_path):
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
    plt.savefig(os.path.join(save_path,f'Diam_HistComp_{img_name}.png'))
    plt.close()
    print(f'Diameter Compare Hist saved under {save_path} for {img_name}')

def boxplot_compare_Bdiameter(diam_pred,diam_annot,img_name,save_path):
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
    plt.savefig(os.path.join(save_path,f'Diam_Boxplot_{img_name}.png'))
    plt.close()
    print(f'Boxplots saved under {save_path} for {img_name}')

def hist_Bdiameter(diam,hbins,img_name, save_path):
    """Plot histogram and probability density of the number of detected bubbles
        (kernel density estimation - Gaussian)
    input: list containing diameters of all bubbles in one image
           bins in histogram; name of img"""
    sns.histplot(diam,bins=hbins,kde=True,color='darkblue',stat='density')
    plt.xlabel('Bubble diameter [mm]')
    plt.ylabel('Density')
    plt.title(img_name)
    plt.savefig(os.path.join(save_path,f'Diam_HistDens_{img_name}.png'))
    plt.close()
    print(f'Diameter Hist+Dens saved under {save_path} for {img_name}')