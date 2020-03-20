# RDGAN
A Pytorch Implementation of RDN, and SRGAN+RDN
## Code Introduction
We have done some experiments on the super-resolution and simultaneously denoising task. One problem for super-resolution and simultaneously denoising task is that the blur after denoising will be amplified by the super-resolution. What we have tried is to utilize GAN to generate some details of the blured part.
We established two super-resolution architecture, one is [RDN]
Codes comprise of three parts, the first part is the 
