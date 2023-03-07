---
layout: page
title: Computer Vision
description: Five projects related to Computer Vision
img: assets/img/hybrid_image.jpg
importance: 1
category: course
---

# Convolution and Hybrid Images
## Overview 
To write an image filtering function and use it to create hybrid image based on <a href = "http://olivalab.mit.edu/publications/OlivaTorralb_Hybrid_Siggraph06.pdf">paper</a>. Hybrid images are static images that change in interpretation as a function of the viewing distance. To create a hybrid image, I need to blend the high frequency portion of one image with the low-frequency portion of another

## Implement Convolution with NumPy
### Gaussian Kernels
Gaussian filters are generally used to blur the images. In this step, I implemented the Gaussian kernels as shown below.<br/>
<img src="/assets/img/2dGaussian.png"  width="40%" height="40%">
### Convolution
Convolution (or filtering) is a fundamental image processing tool. In this section, I implemented `my_conv2d_numpy()` based on the equation 

$$h[m,n]=\sum f[k,l]I[m+k,n+l]$$

where `f=filter (size k x l)I = image (size m x n)`, In `my_conv2d_numpy()`, I firstly do a padding for the input image according to the the `filter.shape[0]//2` and `filter.shape[1]//2` to ensure the size of the image would not change after filtering. 
Then, I use nested for loop to do the element-wise multiplication between the filter and image. Then, take the sum of the multiplication as the result for the value in that index



# SIFT Local Feature Matching

# Camera Calibration and Fundamental Matrix Estimation with RANSAC

# Scene Recognition with Deep Learning

# Semantic Segmentation Deep Learning