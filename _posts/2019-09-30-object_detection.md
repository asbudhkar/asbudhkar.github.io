---
title: "Object Detection using Faster RCNN and SORT"
data: 2019-09-30
tags: [computer vision, deep learning]
header:
    excerpt: "Object Detection"
---
<p class="aligncenter">
    <img src="/images/faster-RCNN.png" width="300" height="150"/>
</p>

<style>
.aligncenter {
    text-align: center;
}
</style>

Object recognition techniques are studied increasingly due to their applications in video analytics and understanding images. Traditional methods include handcrafted feature generation and shallow networks for training. Deep learning architectures can learn high-level deeper features and thus showed better performance compared to the traditional methods. Object segmentation and tracking methods are widely researched in Computer Vision community due to their vast range of applications. The applications include face detection, medical imaging, video-based surveillance systems, self-driving vehicles, etc. A number of challenges related to obstruction, motion blur, deformation need to be handled while solving problems. Heterogenous objects, interacting objects make it difficult to segment and track objects. Quick movement handling
and real time processing are required in some applications which require design of application specific algorithms. 

Used Faster RCNN and SORT for object detection and tracking and designed a computer vision application to detect objects in people’s hands in videos with applications in surveillance systems, robotics and inventory management system.

Simple online and realtime tracking is simple tracking algorithm that performs Kalman filtering and a method to measure bounding box overlap. It can track multiple objects in realtime.

<a href="https://github.com/asbudhkar/Object-recognition-in-Videos">(Link to Project)</a>