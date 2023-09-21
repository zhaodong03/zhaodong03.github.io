---
layout: page
title: Convolution and Hybrid Images
description: 
img: assets/img/hybrid_image.jpg
importance: 1
category: course
---

# Table of Contents
- [Table of Contents](#table-of-contents)
- [Convolution and Hybrid Images](#convolution-and-hybrid-images)
  - [Overview](#overview)
  - [Implement Convolution with NumPy](#implement-convolution-with-numpy)
    - [Gaussian Kernels](#gaussian-kernels)
    - [Convolution](#convolution)
    - [Hybrid Images](#hybrid-images)
  - [Frequency Domain](#frequency-domain)
  - [Frequency Domain Convolutions](#frequency-domain-convolutions)
  - [Frequency Domain Deconvolutions](#frequency-domain-deconvolutions)
    - [Limitation of Deconvolutions](#limitation-of-deconvolutions)

# Convolution and Hybrid Images
## Overview 
To write an image filtering function and use it to create hybrid image based on <a href = "http://olivalab.mit.edu/publications/OlivaTorralb_Hybrid_Siggraph06.pdf">paper</a>. Hybrid images are static images that change in interpretation as a function of the viewing distance. To create a hybrid image, I need to blend the high frequency portion of one image with the low-frequency portion of another

## Implement Convolution with NumPy
### Gaussian Kernels
Gaussian filters are generally used to blur the images. In this step, I implemented the Gaussian kernels as shown below.<br/>
<img src="/assets/img/cv/2dGaussian.png"  width="40%" height="40%">

After applying this Gaussain filter to an image, it blurs this image
<img src="/assets/img/cv/after_2dGaussian.png"  width="100%">

### Convolution
Convolution (or filtering) is a fundamental image processing tool. In this section, I implemented `my_conv2d_numpy()` based on the equation 

$$h[m,n]=\sum f[k,l]I[m+k,n+l]$$

where `f=filter (size k x l)I = image (size m x n)`, In `my_conv2d_numpy()`, I firstly do a padding for the input image according to the the `filter.shape[0]//2` and `filter.shape[1]//2` to ensure the size of the image would not change after filtering. 
Then, I use nested for loop to do the element-wise multiplication between the filter and image. Then, take the sum of the multiplication as the result for the value in that index

Below, there are some exmaple of differents kernels <br/>

|<img width=385/>|<img width=385/>|
| :------: | :------: |
| <img src="/assets/img/cv/identity.png" width="80%" > | <img src="/assets/img/cv/box.png" width="80%"> |
| Identity Filter | Box Filter | 
| <img src="/assets/img/cv/sobel.png" width="80%"> | <img src="/assets/img/cv/discrete.png" width="80%"> |
| Sobel Filter | Discrete Laplacian filter Filter | 

### Hybrid Images
Hybrid image (images give you different view when the distance changes) is the sum of a low-pass filtered version of one image and a high-pass filtered version of another image. To get the hybrid image, I do the following three steps
* Step 1: Get the low frequencies of image1 Using `my_conv2d_numpy(image1, filter) `
* Step 2: Get the high frequencies of image2 by subtracting the low frequencies from the original images
* Step 3: Get the hybrid image by add the low frequencies of image 1 and the high frequencies of image 2 (Note: we need to clip the final image to make sure that the value is in the proper range [0,1], using the `numpy.clip()` )

Below are some examples:
<img src="/assets/img/cv/hybrid_image_scales.jpg" width="100%" >

<img src="/assets/img/cv/hybrid_image_scales_2.jpg" width="100%" >

<img src="/assets/img/cv/hybrid_image_scales_3.jpg" width="100%" >

<img src="/assets/img/cv/hybrid_image_scales_4.jpg" width="100%" >

<img src="/assets/img/cv/hybrid_image_scales_5.jpg" width="100%" >

## Frequency Domain
## Frequency Domain Convolutions
The Fourier transform of the convolution of two functions is the product of their Fourier transforms, and convolution in spatial domain is equivalent to multiplication in frequency domain. Therefore, we could do the convolution in the spatial domain.

I have the orgial image and kernal shown below in Spatial and Frequence Domain seperately.
<img src="/assets/img/cv/dog_frequence.png" width="100%" >

<img src="/assets/img/cv/2dGaussian_frequence.png" width="100%" >

After the multiplication, I get the following result
<img src="/assets/img/cv/dog2_frequence.png" width="100%" >

## Frequency Domain Deconvolutions
The Convolution Theorem refers the convolution in spatial domain is equivalent to the multiplication in frequency domain. With that idea, when comes to the invert the convolution, we could think about the division in the frequency domain. This idea leads to the deconvolution.

For the above example, after the frequency domain division, we get the following the result
<img src="/assets/img/cv/dog3_frequence.png" width="100%" >

### Limitation of Deconvolutions
There are two factors that the deconvalutions mostly does not work in real world
* First factor is that in the real world, we mostly do not know the filter 
* The second factor is that for some filter (like zero filter), it is almost impossible to do deconvolution

Also, the deconvutions are very sensitive to noise.

Here is a example, provided the following image with its kernel
<img src="/assets/img/cv/mystery_example.png" width="100%" >

<img src="/assets/img/cv/mystery_img.png" width="100%" >

<img src="/assets/img/cv/mystery_kernal.png" width="100%" >

We could get the good deconvultion by doing the frequency domain division, shown below
<img src="/assets/img/cv/deconv.png" width="100%" >

However, if we add the salt and pepper noise, we could only get this
<img src="/assets/img/cv/deconv_noise.png" width="100%" >