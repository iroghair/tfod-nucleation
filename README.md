# Bubble Detection with Tensorflow Object Detection
This set of codes is used for the detection of nucleating gas bubbles on images using the Tensorflow Object Detection API. It is based on the [Tensorflow Object Detection course by Nicolas Renotte](https://github.com/nicknochnack/TFODCourse). With the provided code one can train a Convolutional Neural Network (CNN) from the Tensorflow Model Zoo with custom images. Afterwards, the custom network can be used to detect bubbles on experimental images and evaluate important characteristics, such as the number of instances in an image, average bubble size, etc. The code also includes a bubble tracker used for detecting moving and detaching bubbles.


## Main Codes
* 1_Train_CNN_from_TFOD_Zoo.ipynb</b> Executes transfer learning with pretrained model from Tensorflow model Zoo
* 2_Test_CNN_on_new_imgs.ipynb</b> Get evaluation metrics of custom network on new images
* 3_Detection_and_Analysis.py</b> Make and visualize detections using the custom model on (experimental) images. Calculate and plot several bounding box/bubble characteristics.
* 4_BubbleTracker.py</b> Kalman Filter based Centroid Tracker for identifying dissapearing bubbles from image n to n+1 within a time series (experimental images from nucleation process)
* 5_Retrain_custom_CNN.ipynb</b> Train custom model again with new images.

## More info
Read HowTo_BubTFOD

Diploma Thesis by Josefine Gatter: Characterization of the nucleation process for supersaturation-induced gas evolution with a Deep Learning-based bubble detection method (2022)
