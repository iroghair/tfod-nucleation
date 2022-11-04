# Setting up TensorFlow

Used Model from Tensorflow 2 Model Zoo:
https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf2_detection_zoo.md

>>>>> SSD MobileNet V2 FPNLite 320x320 <<<<<<

Tensorflow Object Detection API

Set Up
1) Installing Tensorflow
- error: no distribution for installing tensorflow
        check python version in virtual env tfod
                (tfod) $ python --version
        initially latest python version (3.10) installed
        change to python version 3.9
                (tfod) $ conda install python=3.9
        trying to install tensorflow again
                (tfod) $ pip install --upgrade tensorflow

- error: No module named 'tensorflow' (even though commands in jupyter notebook were executed)
        !pip install tensorflow --upgrade
        - for the case that GPU is used (see below), run: !pip install tensorflow-gpu --upgrade
        note: sometimes this only executes in the second try

- error: No module named 'object_detection'
        !pip install tensorflow-object-detection-api

- error: module 'tensorflow' has no attribute 'contrib'
        TRY #1: didn't work
                import tensorflow.compat.v1 as tf
                tf.disable_v2_behavior()
        TRY #2: didn't work
        - check python version in jupyter notebook
                !python -V
        - check compability: https://www.tensorflow.org/install/pip#package-location
                !pip install --upgrade https://storage.googleapis.com/tensorflow/windows/gpu/tensorflow_gpu-2.6.0-cp38-cp38-win_amd64.whl
        - run: !pip install tensorflow-gpu==1.14

2) Installing missing packages
        Check with "pip list"

        pip install matplotlib
        pip install pyyaml

Working with 2. Training and Detection.ipynb
1) Opening file with "jupyter notebook"
- import error: cannot import name 'constants' from partially initialized module 'zmq.backend.cython' (most likely due to a circular import)
        pip uninstall pyzmq
        pip install pyzmq
        --> can be opened again

2) Make sure to use "tfod" kernel!

3) Verify installation
- error: Module 'tensorflow' has no attribute 'contrib'

--> verify tensorflow installation process
        https://www.youtube.com/watch?v=dZh_ps8gKgs
        -install protoc: download from
        -extract and add protoc/bin to Path in advanced system settings
        -Anaconda prompt: cd to models/research directory and call:
                protoc object_detection/protos/*.proto --python_out=.
        - copy models/research/object_detection/packages/tf2 --> setup.py into models/research
        - from model/research run:
                python -m pip install .
        --> solved the problem: VERIFICATION SCRIPT in jupyter notebook gives back "OK"

4) Download Model from Tensorflow Model Zoo and configure to personal case via notebook
        in C:\Users\20214373\Projects\TfObjectDetection\TFODCourse\Tensorflow\workspace\pre-trained-models

EVERYTHING WORKED UNTIL TRAIN MODEL
further steps:
        - install CUDA
        - install CuDNN
        - get labeled images / make labelImg work
        - train model

################################
in case that PC got a Nvidia GPU --> accelerate Training with CUDA & cuDNN

INSTALLING CUDA CuDNN

1) check tensorflow version with
        (tfod) $ pip list
        here 2.6.0
2) got to https://www.tensorflow.org/install/source_windows (for windows, also available for mac..)
        --> GPU
        check required cuDNN and CUDA versions (here: cuDNN 8.1; CUDA 11.2)
3) use websites to download, considering personal target platform
        CUDA
        https://developer.nvidia.com/cuda-11.2.0-download-archive?target_os=Windows&target_arch=x86_64&target_version=10&target_type=exenetwork

        cuDNN
        https://developer.nvidia.com/rdp/cudnn-archive
        needs nvidia account
        extract all

4) copy cuDNN files into CUDA directory
        cuDNN bin files (here under: C:\Users\20214373\AdditionalPackages\cudnn-11.2-windows-x64-v8.1.1.33\cuda\bin)
        into CUDA bin folder (here: C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.2\bin)

        same for "include" and "lib/x68" folder

5) add to path
        This PC --> properties --> advanced systems settings --> env variables --> Path
        add following directories to path:

        C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.2\bin
        C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.2\libnvvp
        C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.2\extras\CUPTI\lib64