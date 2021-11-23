import os

import tensorflow as tf
from tensorboard import program

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.python.framework import tensor_util
from IPython.display import display

from MetricPlots import precision_plot, recall_plot
from tidecv import TIDE, datasets

# set model name
custom_models = ['my_ssd_mobnet']#,
                #'my_faster_rcnn_resnet50',
                #'my_faster_rcnn_resnet50_2',
                #'my_faster_rcnn_resnet50_3',
                #'my_faster_rcnn_resnet50_4',
                #'my_faster_rcnn_resnet50_5']

# models path
models_path = os.path.join('Tensorflow','workspace','models')


def set_paths(model_name):
    """Get path to train and eval folders of specific model"""
    custom_model_path = os.path.join('Tensorflow','workspace','models',model_name)
    train_path = os.path.join(custom_model_path,'train')
    eval_path = os.path.join(custom_model_path,'eval')
    return train_path, eval_path

def get_event_file(epath):
    """Get path to event file (train or eval)"""
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
    #print(event_df)
    event_dict[model_name] = event_df
    return event_dict


# START PROGRAM

train_dict = {}
eval_dict = {}


for model in custom_models:
    train_path, eval_path = set_paths(model)
    # train event files
    train_efile_path = get_event_file(train_path)
    train_metrics_dict = get_evaluation_metrics(model,train_efile_path,train_dict)
    # evaluation event files
    eval_efile_path = get_event_file(eval_path)
    eval_metrics_dict = get_evaluation_metrics(model,eval_efile_path,eval_dict)

tide = TIDE()
tide.evaluate(datasets.COCO(), datasets.COCOResult(eval_efile_path), mode=TIDE.BOX)
tide.summarize()  # Summarize the results as tables in the console
tide.plot()       # Show a summary figure

#edict_keys = list(train_metrics_dict.keys())

#for model in edict_keys:
#    metrics = eval_metrics_dict[model].iloc[10:21]

precision_fig = precision_plot(eval_metrics_dict)
recall_ig = recall_plot(eval_metrics_dict)
plt.show()

#launch_tensorboard(train_path)
launch_tensorboard(eval_path)

# TODO for train_metrics:
# separate df into Value.unique() to separate losses from each other

# for view in debug console:
#display(event_df)

x=1