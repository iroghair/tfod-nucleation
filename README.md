# Bubble Detection with Tensorflow Object Detection
<p>This set of codes is used for gas bubble detection in images using the Tensorflow Object Detection API. It is based on the Tensorflow Object Detection course by Nicolas Renotte (https://github.com/nicknochnack/TFODCourse). With the provided code one can train a Convolutional Neural Network (CNN) from the Tensorflow Model Zoo with custom images. Afterwards, the custom network can be used to detect bubbles on experimental images and evaluate important characteristics, such as nunmber of instances in an image, average bubble size, etc. The code also includes a bubble tracker used for detecting disappearing bubbles.


## Main Codes
<br />
<b>1_Train_CNN_from_TFOD_Zoo.ipynb.</b> Executes transfer learning with pretrained model from Tensorflow model Zoo
<br/><br/>
<b>2_Test_CNN_on_new_imgs.ipynb.</b> 5)	Get evaluation metrics of custom network on new images
<br/>
<b>3_Detection_and_Analysis.py.</b> Make and visualize detections using the custom model on (experimental) images. Calculate and plot several bounding box/bubble characteristics.
<br/>
<b>4_BubbleTracker.py.</b> Kalman Filter based Centroid Tracker for identifying dissapearing bubbles from image n to n+1 within a time series (experimental images from nucleation process)
<br/>
<b>5_Retrain_custom_CNN.ipynb.</b> Train custom model again with new images.
<br />