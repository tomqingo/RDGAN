# RDGAN
A Pytorch Implementation of RDN, and SRGAN+RDN
## Code Introduction
We have done some experiments on the super-resolution and simultaneously denoising task. One problem for super-resolution and simultaneously denoising task is that the blur after denoising will be amplified by the super-resolution. What we have tried is to utilize GAN to generate some details of the blured part.  
We established two super-resolution architecture, one is [RDN](http://openaccess.thecvf.com/content_cvpr_2018/papers/Zhang_Residual_Dense_Network_CVPR_2018_paper.pdf), and the other is [SRGAN](http://openaccess.thecvf.com/content_cvpr_2017/papers/Ledig_Photo-Realistic_Single_Image_CVPR_2017_paper.pdf). A small difference is that the generator of SRGAN is RDN, taking advantage of superior effect of super resolution of RDN.  
Codes comprise of three parts, the first part is the data generator. 5 small patches are randomly selected from DIV2K training datasets, totally 4000 patches. And whole image is preserved for DIV2K validation dataset. The second part is model. The third part is training and testing code, among which train.py is the training code of SRGAN+RDN, train_rdn.py is the training code of RDN. And test_image.py is the testing code of these models.
## Results
Bicubic Downsample x3 RDN
Dataset | PSNR(dB) | SSIM | BRISQUE
--------- | ------------- | ------------- | -------------
Set5 | 35.0033 | 0.9467 | 32.826
Set14 | 31.1504 | 0.8822 | 37.7410  

Bicubic Downsample x3 + Gaussian Noise Level 25 RDN
Dataset | PSNR(dB) | SSIM | BRISQUE
--------- | ------------- | ------------- | -------------
Set5 | 29.3928 | 0.8418 | 47.2392
Set14 | 27.4011 | 0.7577 | 46.4806


Bicubic Downsample x3 + Gaussian Noise Level 25 SRGAN+RDN
Dataset | PSNR(dB) | SSIM | BRISQUE
--------- | ------------- | ------------- | -------------
Set5 | 29.3432 | 0.8392 | 47.4282
Set14 | 27.3637 | 0.7534 | 44.2279  

PSNR and SSIM are calculated by the Y channel of YCrCb image converted from RGB image.  

Image Result  
![](https://github.com/tomqingo/RDGAN/blob/master/images/HR%20image.png)  
![](https://github.com/tomqingo/RDGAN/blob/master/images/LR%20image.png)  
![](https://github.com/tomqingo/RDGAN/blob/master/images/RDN output%20image.png)  
![](https://github.com/tomqingo/RDGAN/blob/master/images/SRGAN+RDN output%20image.png)


