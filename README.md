# DiseasedCellSegmentation_CNN
BITS F312 - Neural Networks and Fuzzy Logic Final Project [Spring 2017] 
A convolutional neural network capable of Diseased Red Blood Cell (RBC) image segmentation

The problem:
Given variable sized images of blood cells and your task is to use computer vision and machine learning to segment the RBCs. This means assigning 1 to each pixel that makes up a RBC in the image and 0 otherwise (called background).  

Training:
The training data consists of 169 variable sized images. Each image is accompanied by a mask which is the ground truth for that segmentation problem. Please note that each image could have more than 1 RBC.

Test:
The test data consists of variable sized RGB images with each image possibly having more than 1 RBC.


Tools Used:
Keras Deep Learning Library with TensorFlow Backend
OpenCV for image processing

Final Accuracy Score (IoU): 98%
