---
title: "Hand Detection using Pose estimation"
data: 2019-12-30
tags: [computer vision, deep learning]
header:
    excerpt: "Hand Detection"
---

<p class="aligncenter">
    <img src="/images/hand.png" width="300" height="150"/>
</p>

<style>
.aligncenter {
    text-align: center;
}
</style>

Open Pose, a human body keypoint estimator, was used for detecting objects in people’s hands. 
The idea is to use the orientation of hand with keypoints corresponding to shoulder, elbow and wrist of hand to predict the bounding box around hand and use the image within bounding box to predict the category of object in hand. Even if the segmentation technique fails due to obstruction or intersecting objects, the hand keypoints which are visible can lead to assist in identification of object in hand.

Videos of person holding objects were collected to use for detecting objects in hand. Videos were captured with cameras at different angles. Dataset included frames extracted from videos captured and for every frame an annotation text file including the name of sample,
class and four bounding box coordinates. CMU Open Pose model was used to capture keypoints for every frame of video. For creating training dataset, OpenCV was used to visualize the keypoints on the frame and captured the shoulder, elbow and wrist coordinates of hand which holds the object. The frames where hand keypoint coordinates were missing were ignored. Thus, the dataset generated was 3 pairs (x, y) of hand coordinates (shoulder, elbow, wrist) as input to model and bounding box coordinates: bottom left and top right that is 2 – (x, y) pairs from annotation file corresponding to the frame as the ground truth.
The dataset was split into 70 % train and 30 % test set.

A 5-layer neural network was trained to generate the bounding box coordinates for object with 3 pairs of hand coordinates as input from the train dataset. Smooth Mean Square Error i.e. Huber loss was used along with Adam optimizer. With a learning rate of 0.001, the model was trained for 200 epochs.

The model was saved and used a hand detector for next steps. Images of objects under consideration was taken and classes assigned. A pretrained VGG-8 classifier trained on COCO dataset was finetuned using images collected.

The test set was used to generate bounding box coordinates using the saved hand detector model. The image within boxes was cropped and classified using a VGG-8 network finetuned on
objects under consideration. The result was stored as a video with the bounding boxes drawn on the frame along with
the class of the object. 

On visualizing the results using Open CV, the results seemed promising. In case all three shoulder, elbow and wrist were detected properly and with proper alignment, bounding box predictions were accurate.

<a href="https://github.com/asbudhkar/Hand-Detector-with-Pose-Estimation">