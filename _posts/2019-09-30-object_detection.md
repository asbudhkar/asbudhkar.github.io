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

Simple online and realtime tracking is simple tracking algorithm that performs Kalman filtering and a method to measure bounding box overlap. It can track multiple objects in realtime. It associates the detected objects across frames. A detection algorithm is used whose results are used by SORT algorithm to match the detected objects in subsequent frames. Each box has an object id and SORT associates the objects in different frames using simple heuristics like maximizing Intersection over Union between boxes in subsequent frames.

# Problem Statement

Use Faster RCNN and SORT for object detection and tracking and design a computer vision application to detect objects in people’s hands from videos with applications in surveillance systems, robotics and inventory management system.

# Proposal

## Create dataset 
Videos of person capturing objects were collected to use for training and testing. Videos captured with cameras at different angles were collected. Dataset includs frames extracted from videos and for every frame annotation including the name of sample, class and four bounding box coordinates captured using OpenCV.

## Import libraries
```python
from __future__ import print_function
import os
import sys
import utils
import argparse
import numpy as np
from PIL import Image

import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchsummary import summary

os.system("git clone https://github.com/abewley/sort.git")

import transforms as T
from engine import train_one_epoch, evaluate

import cv2
import matplotlib.pyplot as plt
import psutil
import sort 
from sort import *
```

## Create Object dataset
```python
class objDataset(object):

    def __init__(self, root, transforms, classes=None):
        self.root = root
        self.transforms = transforms
        self.classes = classes
        # load all images and dicts

        all_imgs = list(sorted(os.listdir(os.path.join(root, "images"))))
        all_dicts = list(sorted(os.listdir(os.path.join(root, "dicts"))))

        self.imgs = []
        self.dicts = []

        idx = 0
        for sd in all_dicts:
          for si in all_imgs:
            if sd[:-4]==si[:-4]:
              self.imgs.append(si)
              self.dicts.append(sd)
              break

    def __getitem__(self, idx):
        # Get image and dict path
        img_path = os.path.join(self.root, "images", self.imgs[idx])
        dict_path = os.path.join(self.root, "dicts", self.dicts[idx])

        # Open image 
        img = Image.open(img_path).convert("RGB")
        # Open corresponding dictionary 
        f = open(dict_path, 'r')
        lines = f.readlines()
        f.close()
        # Get bounding box coordinates 
        box = [int(s) for s in lines[-1].split()]

        # Get bounding box label  
        label = lines[len(lines)-2].replace('\n','')

        boxes = torch.as_tensor([[box[0], box[1], box[0]+box[2], box[1]+box[3]]], dtype=torch.float32)
        
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
    
        # Get a list of classes 
        label1 = args.classes.split(",") 
        i=len(label1) 
    
        while (i!=0): 
                labels = (label1.index(label)+1)*torch.ones((1,), dtype=torch.int64)    
                i=i-1

        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])

        # suppose all instances are not crowd
        iscrowd = torch.zeros((1,), dtype=torch.int64)
        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd
        if self.transforms is not None:
            img, target = self.transforms(img, target)
        return img, target

    def __len__(self):
        return len(self.imgs)
```

## Return predicted boxes and labels for each object
```python
def get_prediction(model, img_path, threshold, classes, device=device):
  label=classes.split(",")
  LABELS = []
  # First label is background
  LABELS.append("background")

  for i in label:
      LABELS.append(i)  
      
  img = Image.open(img_path)
  transform = T.Compose([T.ToTensor()])
  img = transform(img)

  # Get prediction
  with torch.no_grad():
    pred = model([img.to(device)])

  pred_score = list(pred[0]['scores'].detach().cpu().numpy())
  # Get prediction with score above threshold

  pred_t = [pred_score.index(x) for x in pred_score if x>threshold]

  if pred_t:
    pred_t=pred_t[-1]
  else:
    return [],[]

  print(pred[0]['labels'])
  # Return class and box coordinates

  pred_class = [LABELS[i] for i in list(pred[0]['labels'].cpu().numpy())]
  pred_boxes = [[(i[0], i[1]), (i[2], i[3])] for i in list(pred[0]['boxes'].detach().cpu().numpy())]

  if(pred_t is []):
    return [],[]

  pred_boxes = pred_boxes[:pred_t+1]
  pred_class = pred_class[:pred_t+1]
  return pred_boxes,pred_class
```

## Required for SORT as SORT input is in YOLO output form
```python
def to_yolo_form(pred, pred_t):
  scores = pred[0]['scores'][:pred_t+1].unsqueeze(1)
  return torch.cat([pred[0]['boxes'][:pred_t+1,:], scores, scores, pred[0]['labels'][:pred_t+1].float().unsqueeze(1)],1)
```

## Get fasterRCNN object from torchvision models
```python
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    model.to(device)
    classes = args.classes
    # Create list of classes 
    labels = classes.split(",")      
    num_classes = len(labels)+1  # N class + background

    # Get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features

    # Replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
```

## Train the model
```python
    # Set root - Folder in which images and dists are present  
    root_path = '.'
    dataset = objDataset(root_path, get_transform(train=True), labels)
    dataset_test = objDataset(root_path, get_transform(train=False),labels)
    indices = torch.randperm(len(dataset)).tolist()
    
    # Split in train and test 

    #TODO modify the split as per the number of examples    
    dataset = torch.utils.data.Subset(dataset, indices[:500])
    dataset_test = torch.utils.data.Subset(dataset_test, indices[500:600])

    # Define training and validation data loaders
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True, num_workers=0,
        collate_fn=utils.collate_fn)

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=args.batch_size_test, shuffle=False, num_workers=0,
        collate_fn=utils.collate_fn)

    model.to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)

    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

    num_epochs = args.epochs

    for epoch in range(num_epochs):
        # Train for one epoch, printing every 10 iterations
        train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=10)

        # Update the learning rate
        lr_scheduler.step()

    torch.save(model.state_dict(), 'fasterRNN-hand.pt')
```

<a href="https://github.com/asbudhkar/Object-recognition-in-Videos">Link to Project:</a>