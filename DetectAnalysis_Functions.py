from cProfile import label
import os
import json
import cv2
import imutils
import numpy as np
import pandas as pd
import datetime as dt
from matplotlib import pyplot as plt
import seaborn as sns
from ODE_bubble_growth import get_Rb

"""Functions for the detection of bubbles with custom CNN and subsequent analysis"""

size=15
params = {'legend.fontsize': 'large',
          #'figure.figsize': (20,8),
          'axes.labelsize': size,
          'axes.titlesize': size,
          'xtick.labelsize': size*0.75,
          'ytick.labelsize': size*0.75,
          'font.size': size*0.65,
          'axes.titlepad': 25}
plt.rcParams.update(params)

def get_image(im_path, scaling_factor=1): #, bg_img):
    img = cv2.imread(im_path)
    parent_path = os.path.dirname(im_path)
    img_set = os.path.basename(parent_path)
    # to grayscale and back (black/white with RGB format)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    # adjust contrast and brightness
    #alpha = 2.5 # Contrast control (1.0-3.0)
    #beta = 0 # Brightness control (0-100)
    #img = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
    # subtract background
    #img = cv2.absdiff(img, bg_img)
    #cv2.subtract(img1,img2)
    # no image manipulations needed for artificial test images
    if img_set != 'test':
        # Opening JSON file (contains postprocessing parameters)
        json_path = os.path.join('Tensorflow', 'workspace','images','Img_postproc_params.json')
        f = open(json_path)
        # return JSON object as dictionary
        data = json.load(f)
        # parameters for specific data set
        d = data[img_set]
        rot_ang,x1,y1,w,h = d['rotate'],d['xmin'],d['ymin'],d['width'],d['height']
        # rotate image (+ counterclockwise, - clockwise)
        img = imutils.rotate(img, angle=-rot_ang)
        # crop image
        img = img[y1:y1+h, x1:x1+w]
    # rescaling image (resolution) shape[1]=width, shape[0]=height
    img = cv2.resize(img,(int(img.shape[1] * scaling_factor), int(img.shape[0] * scaling_factor)))
    return img

def exclude_partial_pred(detect_dict,img,abs_dist):
    """Removes bounding boxes from detection dict that touch the image boundaries
    Goal: remove partial detections
    abs_dist = allowed distance [in pixels] of bbox to border of img"""
    part_dict = dict(detect_dict)
    im_height, im_width, channels = img.shape
    xdist_rel = abs_dist/im_width
    ydist_rel = abs_dist/im_height
    boxes = part_dict['detection_boxes']
    scores = part_dict['detection_scores']
    multscores = part_dict['detection_multiclass_scores']
    classes = part_dict['detection_classes']
    strided = part_dict['detection_boxes_strided']
    rows = [] # rows that will be deleted
    for i, box in enumerate(boxes):
        ymin, xmin, ymax, xmax = box[0], box[1], box[2], box[3]
        # exclude bboxes that touch the image borders
        if (xmin<xdist_rel or ymin<ydist_rel or xmax>(1-xdist_rel) or ymax>(1-ydist_rel)):
            # delete respective rows from arrays in dict
            rows.append(i)
            part_dict['num_detections'] -= 1
        else:
            pass
    part_dict['detection_boxes'] = np.delete(boxes, rows, 0)
    part_dict['detection_scores'] = np.delete(scores, rows, 0)
    part_dict['detection_multiclass_scores'] = np.delete(multscores, rows, 0)
    part_dict['detection_classes'] = np.delete(classes, rows, 0)
    part_dict['detection_boxes_strided'] = np.delete(strided, rows, 0)
    return part_dict

def get_absolute_pixels(detect_dict,img):
    """Gets absolute pixel values for relative bounding boxes"""
    abs_dict = dict(detect_dict)
    im_height, im_width, channels = img.shape
    boxes = abs_dict['detection_boxes']
    abs_boxes = []
    for box in boxes:
        ymin, xmin, ymax, xmax = box[0], box[1], box[2], box[3]
        # absolute coordinates of box
        (top, left, bottom, right) = (ymin * im_height, xmin * im_width,
                                      ymax * im_height, xmax * im_width)
        abs_boxes.append((top, left, bottom, right))
    abs_dict['detection_boxes'] = np.array(abs_boxes)
    return abs_dict

def get_pred_thresh(detect_dict,score_thresh,pred=True):
    """Removes entries from detection dict of one image
    that belong to detection score < score_thresh
    pred=True for prediction dicts, since it contains other keys than annotation dict"""
    thresh_dict = dict(detect_dict)
    boxes = thresh_dict['detection_boxes']
    scores = thresh_dict['detection_scores']
    classes = thresh_dict['detection_classes']
    # only for prediction dicts
    if pred:
        multscores = thresh_dict['detection_multiclass_scores']
        strided = thresh_dict['detection_boxes_strided']
    else:
        pass
    rows = [] # rows that will be deleted
    # check detection score
    for i, v in enumerate(boxes):
        if scores[i] < score_thresh:
            rows.append(i)
            if pred:
                thresh_dict['num_detections'] -= 1
            else:
                pass
        else:
            pass
    thresh_dict['detection_boxes'] = np.delete(boxes, rows, 0)
    thresh_dict['detection_scores'] = np.delete(scores, rows, 0)
    thresh_dict['detection_classes'] = np.delete(classes, rows, 0)
    # only for prediciton dicts
    if pred:
        thresh_dict['detection_multiclass_scores'] = np.delete(multscores, rows, 0)
        thresh_dict['detection_boxes_strided'] = np.delete(strided, rows, 0)
    else:
        pass
    return thresh_dict

def get_visualization_colors(detect_dict,scaling_factor,score_thresh,img_path):
    """Sets colors for bounding boxes with min. score of score_thresh
    Smallest bounding boxes obtain a different color"""
    scores = detect_dict["detection_scores"]
    boxes = detect_dict["detection_boxes"]
    thresh_boxes = boxes[scores > score_thresh]
    # 102 green
    colors = np.full(len(thresh_boxes), 102, dtype=int)
    # color for smalles bubbles
    i_smallest = []
    # set area threshold (according to COCO metrics "small"<32^2 pixels)
    #img = cv2.imread(img_path)
    img = get_image(img_path,scaling_factor)
    im_height, im_width, channels = img.shape # pixel values
    area_thresh = 32**2 / (im_height*im_width) # relative area value
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
    colors[i_smallest] = 128
    return colors

def unite_detection_dicts(pdict,adict,score_thresh):
    """Unite detection dicts from prediction and annotation
    Predicted detections thresholded my minimum score"""
    dict={}
    i_thresh = pdict["detection_scores"] > score_thresh
    a1=pdict["detection_boxes"][i_thresh]
    a2=adict["detection_boxes"]
    dict["detection_boxes"] = np.concatenate((a1, a2), axis=0)
    a3=pdict['detection_classes'][i_thresh]
    a4=adict['detection_classes']
    dict['detection_classes']=np.concatenate((a3, a4), axis=0)
    a5=pdict['detection_scores'][i_thresh]
    a6=adict['detection_scores']
    dict['detection_scores']=np.concatenate((a5, a6), axis=0)
    return dict

def get_time_diff_name(img_paths, mode="sec"):
    """Convert image names from "DSC_hhmmss" to time difference in seconds or minutes (mode)"""
    # remove "DSC__" from name
    splits = [n.split('__')[1] for n in img_paths]
    # remove file extension
    abs_times_str = [n.split('.')[0] for n in splits]
    # get datetimes (note: random year added)
    abs_times = pd.to_datetime(abs_times_str,format='%H%M%S') #.to_series()
    start_time = min(abs_times)
    # time differences (timedelta)
    rel_times = [t - start_time for t in abs_times]
    rel_times_tot = [t.total_seconds() for t in rel_times]
    rel_times_tot_str = [str(t)+' s' for t in rel_times_tot]
    if mode=="min":
        # convert to minutes (e.g. for large time differences)
        rel_times_tot = [t/60 for t in rel_times_tot]
        rel_times_tot_str = [str(t)+' min' for t in rel_times_tot]
    return rel_times_tot_str

############## PLOTS ##########################

def plot_Bcount(dict, test_path, save_path):
    """Plot number of detected bubbles over time
    (timestamps must be indicated in image name)
    input: dictionary containing number of detected bubbles
           image names as keys"""
    Bcount = list(dict.values())
    dk = list(dict) # dict keys
    if "t" in dk[0]:
        # remove tail of key string
        splits = [key.split('_')[0] for key in dk]
        # remove "t" in front of time indication
        times_str = [s.split('t')[1] for s in splits]
        # get timestamps as int
        times = [int(t) for t in times_str]
    else:
        times_str = dk
        # get timestamps as int
        times = [round(float(key.split(' ')[0])) for key in times_str]
        if dk[0].split(' ')[1] == 's': # convert sec to min
            times = [t / 60 for t in times]
    plt.plot(times, Bcount, 'o',markersize=6, color='limegreen')
    plt.xlabel('Time [min]')
    plt.ylabel('Bubble count [-]')
    name = os.path.basename(os.path.normpath(test_path))
    plt.title("Total number of bubbles: "+name)
    plt.savefig(os.path.join(save_path,f'BubbleCount.png'),bbox_inches="tight")
    plt.close()
    print(f'Bubble count plot saved under {save_path}')

def plot_avrg_Bdiam(dict, test_path, save_path):
    """Plot average bubble diameter over time
    (timestamps must be indicated in image name)
    input: dictionary containing average bubble diameter
           image names as keys"""
    Bdiam = list(dict.values())
    dk = list(dict) # dict keys
    if "t" in dk[0]:
        # remove tail of key string
        splits = [key.split('_')[0] for key in dk]
        # remove "t" in front of time indication
        times_str = [s.split('t')[1] for s in splits]
        # get timestamps as int
        times = [int(t) for t in times_str]
    else:
        # get timestamps as int
        times = [round(float(key.split(' ')[0])) for key in dk]
        if dk[0].split(' ')[1] == 's': # convert sec to min
            times = [t / 60 for t in times]
    # ODE numerical solution
    R_b, t = get_Rb() # bubble radius [m], time [s]
    t_ODE = t / 60 # time vector [min]
    d_B_ODE_m = R_b * 2 # bubble diameter [m]
    d_B_ODE = d_B_ODE_m * 10**(3) # bubble diameter [mm]
    plt.plot(times, Bdiam, 'o', markersize=6, color='limegreen', label="Experiments")
    plt.plot(t_ODE,d_B_ODE, label="Epstein, Plesset model")
    plt.xlabel('Time [min]')
    plt.ylabel('$D_b$(t) [mm]') #Average bubble diameter 
    name = os.path.basename(os.path.normpath(test_path))
    plt.title(name)
    plt.legend()
    plt.savefig(os.path.join(save_path,f'Avrg_Diam.png'),bbox_inches="tight")
    plt.close()
    print(f'Average diameter plot saved under {save_path}')

def plot_avrg_Bdiam_sqrt(dict, test_path, save_path):
    """Plot average bubble diameter over time
    (timestamps must be indicated in image name)
    input: dictionary containing average bubble diameter
           image names as keys"""
    Bdiam = list(dict.values())
    dk = list(dict) # dict keys
    if "t" in dk[0]:
        # remove tail of key string
        splits = [key.split('_')[0] for key in dk]
        # remove "t" in front of time indication
        times_str = [s.split('t')[1] for s in splits]
        # get timestamps as int
        times = [int(t) for t in times_str]
    else:
        # get timestamps as int
        times = [round(float(key.split(' ')[0])) for key in dk]
        if dk[0].split(' ')[1] == 's': # convert sec to min
            times = [t / 60 for t in times]
    # ODE numerical solution
    R_b, t = get_Rb() # bubble radius [m], time [s]
    t_ODE = t / 60 # time vector [min]
    d_B_ODE_m = R_b * 2 # bubble diameter [m]
    d_B_ODE = d_B_ODE_m * 10**(3) # bubble diameter [mm]
    plt.plot(np.sqrt(times), Bdiam, 'o', markersize=6, color='limegreen', label="Experiments")
    plt.plot(np.sqrt(t_ODE),d_B_ODE, label="Epstein, Plesset model")
    plt.xlabel('\u221At')
    plt.ylabel('$D_b$(t) [mm]')#Average bubble diameter
    name = os.path.basename(os.path.normpath(test_path))
    #plt.title(name)
    plt.legend() #loc='upper left'
    plt.savefig(os.path.join(save_path,f'Avrg_Diam_sqrt.png'),bbox_inches="tight")
    plt.close()
    print(f'Average diameter plot (sqrt) saved under {save_path}')

def hist_compare_Bdiameter(diam_pred,diam_annot,my_bins,img_name,save_path):
    """Plot histogram and probability density of the number of detected bubbles
        (kernel density estimation - Gaussian)
    input: lists containing diameters from prediction and annotation
           bins in histogram; name of img"""
    # plot two histograms in one figure
    sns.histplot(diam_pred,bins=my_bins,kde=True,stat='density',color="green",label="Prediction")
    sns.histplot(diam_annot,bins=my_bins,kde=True,stat='density',color="blue",label="Annotation")
    plt.xlabel('Bubble diameter [mm]')
    plt.ylabel('Probability density')
    plt.title(img_name)
    plt.legend()
    plt.savefig(os.path.join(save_path,f'Diam_HistComp_{img_name}.png'),bbox_inches="tight")
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
    plt.savefig(os.path.join(save_path,f'Diam_Boxplot_{img_name}.png'),bbox_inches="tight")
    plt.close()
    print(f'Boxplots saved under {save_path} for {img_name}')

def hist_Bdiameter(diam,hbins,img_name,save_path):
    """Plot histogram and probability density of the number of detected bubbles
        (kernel density estimation - Gaussian)
    input: list containing diameters of all bubbles in one image
           bins in histogram; name of img"""
    sns.histplot(diam,bins=hbins,kde=True,color='green',stat='density')
    plt.xlabel('Bubble diameter [mm]')
    plt.ylabel('Probability density')
    plt.title("Number of detected bubbles: "+img_name)
    name = img_name.split('.')[0]
    plt.savefig(os.path.join(save_path,f'Diam_HistDens_{name}.png'))
    plt.close()
    print(f'Diameter Hist+Dens saved under {save_path} for {img_name}')

def hist_all_pred_diams(detect_dict,hist_bins,test_path,save_path):
    """HISTOGRAMS OF BUBBLE DIAMETER PREDICTIONS FROM ALL IMAGES
    one subaxis per image"""
    new_keys=[]
    hist_dict = dict(detect_dict)
    old_keys = list(hist_dict)
    # change keys of dict to respective time stamp
    if "t" in old_keys[0]: # for Alessandros images
        for k in old_keys:
            # remove tail of key string
            splits = k.split('_')[0]
            # remove "t" in front of time indication
            times_str = splits.split('t')[1]
            new_keys.append(times_str)
            # replace old key with new key
            hist_dict[times_str] = hist_dict.pop(k)
    else:
        for k in old_keys: # for new images
            times_str = k.split(' ')[0]
            new_keys.append(times_str)
            hist_dict[times_str] = hist_dict.pop(k)
    # sort keys by timestamp
    new_keys.sort(key=float)
    # intialize figure
    fig, ((ax1,ax2,ax3), (ax4,ax5,ax6), (ax7,ax8,ax9)) = plt.subplots(3, 3, sharex=True, sharey=True)
    name = os.path.basename(os.path.normpath(test_path))
    fig.suptitle(name)
    ax_names = [ax1,ax2,ax3,ax4,ax5,ax6,ax7,ax8,ax9]
    plt.rcParams.update({'axes.titlesize':'small'})
    plt.rcParams.update({'axes.titlepad':1})
    #sns.set(font_scale=1)
    # plot dict entries in individual histograms
    for i,nk in enumerate(new_keys):
        sns.histplot(hist_dict[nk],ax=ax_names[i],bins=hist_bins,kde=True,stat='density',color='darkblue')
        ax_names[i].set_title(("t = "+nk+" min"))
        ax_names[i].set(ylabel=None)
    #plt.tick_params(labelcolor="none", bottom=False, left=False)
    fig.text(0.5, 0.01, 'Bubble diameter [mm]', ha='center', va='center', fontsize=size*0.75) #0.5,0.04
    fig.text(0.06, 0.5, 'Probability density', ha='center', va='center', rotation='vertical',fontsize=size*0.75) #0.08,0.5
    #plt.tight_layout()
    # spacing between subplots
    plt.subplots_adjust(wspace=0.1,hspace=0.2)
    plt.savefig(os.path.join(save_path,f'All_Diams_Hist.png'),bbox_inches="tight")
    plt.close()
    print(f'Joint Diameter Hist saved under {save_path}.')

def Bdiams_over_t(dict,save_path):
    new_keys=[]
    old_keys = list(dict)
    # change keys of dict to respective time stamp
    for k in old_keys:
        if "t" in k: # for allessandros pics
            # remove tail of key string
            splits = k.split('_')[0]
            # remove "t" in front of time indication
            times_str = splits.split('t')[1]
        else:
            splits = k.split(' ')[0] # remove unit
            times_str = round(float(splits))
            if k.split(' ')[1] == 's': # convert sec to min
                times_str = str(round(times_str / 60))
                #times_str = str(times_str / 60)
        new_keys.append(times_str)
        # replace old key with new key
        dict[times_str] = dict.pop(k)
    # sort keys by timestamp
    lists = sorted(dict.items()) # sorted by key, return a list of tuples
    x, y = zip(*lists) # unpack a list of pairs into two tuples
    for xe, ye in lists:
        plt.scatter([xe]*len(ye), ye, c='r', s=20)
    plt.xlabel('Time [min]')
    plt.ylabel('Bubble diameter [mm]')
    plt.savefig(os.path.join(save_path,f'All_BDiams_t.png'),bbox_inches="tight")
    plt.close()
    print(f'All Diameters over time-plot saved under {save_path}.')
