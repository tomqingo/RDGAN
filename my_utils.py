import math
import torch
import torch.nn as nn
import numpy as np
from skimage.measure import compare_psnr, compare_ssim 

def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.kaiming_normal(m.weight.data, a=0, mode='fan_in')
    elif classname.find('Linear') != -1:
        nn.init.kaiming_normal(m.weight.data, a=0, mode='fan_in')
    elif classname.find('BatchNorm') != -1:
        # nn.init.uniform(m.weight.data, 1.0, 0.02)
        m.weight.data.normal_(mean=0, std=math.sqrt(2./9./64.)).clamp_(-0.025,0.025)
        nn.init.constant(m.bias.data, 0.0)

def calc_psnr(img1, img2, max=255.0):
    return 10*np.log10(((max ** 2) / ((img1 - img2) ** 2).mean()))

def convert_rgb_to_y(img, dim_order='hwc'):
    if dim_order == 'hwc':
        return 16. + (64.738 * img[..., 0] + 129.057 * img[..., 1] + 25.064 * img[..., 2]) / 256.
    else:
        return 16. + (64.738 * img[0] + 129.057 * img[1] + 25.064 * img[2]) / 256.

def denormalize(img):
    img = img*255.0
    img = np.clip(img, 0, 255.0)
    return img

def batch_PSNR(img, imclean, scale, data_range):
    Img = img.data.cpu().numpy().astype(np.float32)
#    Iclean = imclean.data.cpu().numpy().astype(np.float32)
    Iclean = imclean.data.cpu().numpy().astype(np.float32)
    PSNR = 0
    SSIM = 0.
    for i in range(Img.shape[0]):
        img_org = Img[i,:,:,:]
        img_clean = Iclean[i,:,:,:]
#        img_org_denor = denormalize(img_org)
#        img_clean_denor = denormalize(img_clean)
        img_org = convert_rgb_to_y(denormalize(img_org), dim_order='chw')
        img_clean = convert_rgb_to_y(denormalize(img_clean), dim_order='chw')
        img_org = img_org[scale:-scale, scale:-scale]
        img_clean = img_clean[scale:-scale, scale:-scale]
        PSNR += calc_psnr(img_org, img_clean)
#        PSNR += compare_psnr(np.transpose(Iclean[i,:,:,:],(1,2,0)), np.transpose(Img[i,:,:,:],(1,2,0)), data_range=data_range)
#        SSIM += compare_ssim(np.transpose(Iclean[i,:,:,:],(1,2,0)), np.transpose(Img[i,:,:,:],(1,2,0)), data_range=1.0, multichannel=True)
        SSIM += compare_ssim(img_org, img_clean, data_range=data_range)
    return (PSNR/Img.shape[0], SSIM/Img.shape[0])

def data_augmentation(image, mode):
    out = np.transpose(image, (1,2,0))
    if mode == 0:
        # original
        out = out
    elif mode == 1:
        # flip up and down
        out = np.flipud(out)
    elif mode == 2:
        # rotate counterwise 90 degree
        out = np.rot90(out)
    elif mode == 3:
        # rotate 90 degree and flip up and down
        out = np.rot90(out)
        out = np.flipud(out)
    elif mode == 4:
        # rotate 180 degree
        out = np.rot90(out, k=2)
    elif mode == 5:
        # rotate 180 degree and flip
        out = np.rot90(out, k=2)
        out = np.flipud(out)
    elif mode == 6:
        # rotate 270 degree
        out = np.rot90(out, k=3)
    elif mode == 7:
        # rotate 270 degree and flip
        out = np.rot90(out, k=3)
        out = np.flipud(out)
    return np.transpose(out, (2,0,1))
