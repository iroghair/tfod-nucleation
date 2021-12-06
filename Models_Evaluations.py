import os

import tensorflow as tf
from tensorboard import program

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from openpyxl import load_workbook
from tensorflow.python.framework import tensor_util
from IPython.display import display

from MetricPlots import precision_plot, recall_plot, plot_total_loss, plot_learningrate
from tidecv import TIDE, datasets

"""Get properties that are available in Tensorboard for custom Object Detection models"""

# set model name
custom_models = [#'my_faster_rcnn_resnet101_v1_640',
                'my_faster_rcnn_resnet101_v1_1024_2',
                'my_faster_rcnn_resnet152_v1_1024',
                #'my_ssd_resnet50_v1_fpn_1024',
                #'my_ssd_resnet101_v1_fpn_1024',
                'my_centernet_hg104_512',
                'my_centernet_hg104_1024',
                #'my_centernet_hg104_1024_2',
                #'my_centernet_hg104_1024_3',
                'my_centernet_resnet101_v1_fpn_512',
                'my_efficientdet_d1']
                #'my_efficientdet_d4']
                #'my_faster_rcnn_resnet101_v1_1024']
                #'my_faster_rcnn_resnet101_v1_640']
                #'my_faster_rcnn_resnet101_v1_1024_3']

# models path
models_path = os.path.join('Tensorflow','workspace','models')
images_path = os.path.join('Tensorflow','workspace','images')


def set_paths(model_name):
    """Get path to train and eval folders of specific model"""
    custom_model_path = os.path.join('Tensorflow','workspace','models',model_name)
    train_path = os.path.join(custom_model_path,'train')
    eval_path = os.path.join(custom_model_path,'eval')
    return train_path, eval_path

def get_event_file(epath):
    """Get path to event file (train or eval)"""
    # Make sure that only one file exists in path
    event_file = os.listdir(epath)
    event_path = os.path.join(epath,event_file[0])
    return event_path

def launch_tensorboard(event_path):
    """Launches Tensorboard for graphical evaluation"""
    tb = program.TensorBoard()
    tb.configure(argv=[None, '--logdir', event_path])
    url = tb.launch()
    print(f"Tensorflow listening on {url}")

def get_evaluation_metrics(model_name, efile_path, event_dict):
    event_df = pd.DataFrame(columns=['Value', 'Step', 't', 't_type'])
    for event in tf.compat.v1.train.summary_iterator(efile_path):
        for value in event.summary.value:
            t = tensor_util.MakeNdarray(value.tensor)
            df = pd.DataFrame([[value.tag, event.step, t, type(t)]], columns=['Value', 'Step', 't', 't_type'])
            #print(value.tag, event.step, t, type(t))
            event_df = event_df.append(df, ignore_index=True)
    print(model_name, event_df, sep='\n')
    event_dict[model_name] = event_df
    return event_dict

#######################
# START PROGRAM
#######################

train_dict = {}
eval_dict = {}

for model in custom_models:
    train_path, eval_path = set_paths(model)
    # train event files
    train_efile_path = get_event_file(train_path)
    train_dict = get_evaluation_metrics(model,train_efile_path,train_dict)
    # evaluation event files
    eval_efile_path = get_event_file(eval_path)
    eval_dict = get_evaluation_metrics(model,eval_efile_path,eval_dict)

# save dicts to excel
def save_dict_to_excel(dict,path):
    writer=pd.ExcelWriter(path, engine='openpyxl')
    # save content of one key into one sheet
    for model in dict.keys():
        df = dict[model]
        name = str(model)
        # only 31 characters for sheet name allowed
        if len(name) > 31:
            count = len(name) - 31
            # shorten name from the front
            name = name[count:]
        df.to_excel(writer, sheet_name=name)
    writer.save()
    writer.close()

save_dict_to_excel(train_dict,"Train_Metrics_Dict.xlsx")
save_dict_to_excel(eval_dict,"Eval_Metrics_Dict.xlsx")

# compare precisions of custom models
precision_fig = precision_plot(eval_dict)
plt.savefig(os.path.join(models_path,"Precisions_Compare_allfromTUe.png"))
plt.close()
# compare recalls of custom models
recall_ig = recall_plot(eval_dict)
plt.savefig(os.path.join(models_path,"Recalls_Compare_allfromTUe.png"))
plt.close()
print("Precision and Recall plot saved under 'Workspace/Models'")

# plot Training metrics
tot_loss_dict = {}
learn_rate_dict = {}
for model in custom_models:
    train_prop = train_dict[model]
    tot_loss_dict[model] = train_prop[train_prop.Value=='Loss/total_loss']
    learn_rate_dict[model] = train_prop[train_prop.Value=='learning_rate']
    # plot and save total loss figure
    plot_total_loss(tot_loss_dict[model],os.path.join(models_path,model))
    # plot and save learning rate figure
    plot_learningrate(learn_rate_dict[model],os.path.join(models_path,model))

#launch_tensorboard(train_path)
#launch_tensorboard(eval_path)
# for view in debug console:
#display(event_df)

# edict_keys = list(eval_metrics_dict.keys())

# TODO how to implement TIDE
if 0:
    tide = TIDE()
    tide.evaluate(datasets.COCO(), datasets.COCOResult(eval_efile_path), mode=TIDE.BOX)
    tide.summarize()  # Summarize the results as tables in the console
    tide.plot()       # Show a summary figure