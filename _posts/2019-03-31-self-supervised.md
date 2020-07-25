---
title: "Image classification using Split-brain Autoencoder"
data: 2019-03-31
tags: [deep learning, computer vision]
header:
    excerpt: "Split-brain Autoencoder"
---
<img src="/images/autoencoder3.png" width="300" height="150"/>
<img src="/images/autoencoder.png" width="300" height="150"/>
<img src="/images/autoencoder2.png" width="300" height="150"/>

Deep learning algorithms have shown that, when given large collections of labelled data, they can achieve human-level performance on computer vision tasks. However, for many practical tasks, the availability of data is limited. Self-supervised pretraining is a method of training whereby a network predicts a part of its input using an another unseen part, which acts as the label. The objective is to learn useful representations of the data in order to fine-tune with supervision on downstream tasks such as image classification.

Split-Brain Autoencoder method finds useful global features for classification by solving complementary prediction tasks and therefore utilizing all data in the input. The network is divided into two fully convolutional sub-networks and each is trained to predict one subset of channels of input from the other. For fine-tuning, a classifier is added as the last layer. Using a dataset of 96x96 images, with 512k unlabeled images, 64k labelled training images, and 64k labelled validation images, we perform
1000-class classification.

The approach consists of splitting an image into two subsets of input channels (2 to 1 for a 3-channel space), preferably using a color space that separates color and luminosity. It then passes each subset through a fully convolutional architecture in order to predict the other subset. To make this prediction, it takes the Cross
Entropy loss between the network output and a downsampled, quantized version of the original image (acting as labels). To clarify this with our numbers, the input image has 96x96 input features and each
sub-network has 12x12 output features, each of which corresponds to a pixel in a 12x12 downsampled ground truth of the input image. The number of output channels in each sub-network corresponds to the number of classes for each pixel, which is exactly the number of colors into which each channel was quantized into. Fine-tuning consists of adding a classifier on top of the concatenated output of
the two sub-networks.