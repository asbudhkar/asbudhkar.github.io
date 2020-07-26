---
title: "Inclusive Images"
data: 2018-12-31
tags: [computer vision, deep learning]
header:
    excerpt: "Inclusive images Kaggle challenge"
---
<p class="aligncenter">
    <img src="/images/inclusive_images.png" width="300" height="150"/>
</p>

<style>
.aligncenter {
    text-align: center;
}
</style>
Modern convolutional networks are not known to be robust against distributional skew, i.e. generalizing over different distributions in testing vs. training. However, in real world systems, training data is not inclusive enough of all possible scenarios, and our prediction models usually tend to fail. In this project, attempt is made to try out approaches to train classifiers to obtain good results over different geographical distributions

One area of interest where we would like neural networks to generalize especially well is across geographical distributions. It is understood that a doctor’s clinic in the United States or in Europe is unlikely to resemble a doctor’s clinic in India or in Africa, proving to be a problem for our classifiers’ predictions. The weights that the classifier will learn for the clinic in the US will be very different from those it would learn from a clinic in Africa, and hence it would have a hard time predicting ”clinic” for the latter. To tackle this problem, one needs to be able to model our classifiers to perform well on geographical stress tests.

Baseline image recognition machine leaning models Densenet and Resnet were tried. To account for the inconsistent distributions, weighted sampling  an importance sampling technique that is a variant of stratified sampling  was used to train on data more efficiently. In essence, weighted sampling takes an input and scales it by the weights that it has been given. Then the sum of the weights is divided by the number of classes, and these equal partitions are used to sample from randomly, and choose the input it aligns to in the original distribution. . After weighted sampling there was a huge boost in performance observed.

<a href="https://github.com/asbudhkar/Kaggle-Inclusive-Images-Challenge">(Link to Project)</a>