import argparse
import time

import torch
#from PIL import Image
import PIL.Image as pil_image
from torch.autograd import Variable
from torchvision.transforms import ToTensor, ToPILImage
from my_utils import *

from model_rdn import Generator_RDN
from dataset import bicubic_downsample, modcrop, bicubic_sample
import os
import numpy as np
import cv2
from skimage.measure import compare_psnr, compare_ssim
from skimage.io import imsave
import torch.nn as nn
import numpy as np

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0,1"

parser = argparse.ArgumentParser(description='Test Single Image')
parser.add_argument('--upscale_factor', default=3, type=int, help='super resolution upscale factor')
parser.add_argument('--test_mode', default='GPU', type=str, choices=['GPU', 'CPU'], help='using GPU or CPU')
parser.add_argument('--image_set', type=str, default='data/Set5', help='test low resolution image name')
parser.add_argument('--model_name', default='rdn_final_149.pth', type=str, help='generator model epoch name')
parser.add_argument('--test_noiseL',type=float, default=25, help='noise level used on test set')
parser.add_argument('--save_path', type=str, default='./bicubic_denoise/Set5', help='Path to save the result')
parser.add_argument('--G0', type=int, default=64, help='default number of filters')
parser.add_argument('--RDNkSize', type=int, default=3, help='default kernel size')
parser.add_argument('--RDNconfig', type=str, default='B', help='parameters config of RDN')
parser.add_argument('--n_colors',type=int, default=3, help='channels')

opt = parser.parse_args()


UPSCALE_FACTOR = opt.upscale_factor
TEST_MODE = True if opt.test_mode == 'GPU' else False
IMAGE_SET = opt.image_set
MODEL_NAME = opt.model_name
TEST_NOISE_LEVEL = opt.test_noiseL
SAVE_PATH = opt.save_path

model = Generator_RDN(opt)
if TEST_MODE:
    model.cuda()
    model.load_state_dict(torch.load('DIV/logs_srdn_rdn/model/' + MODEL_NAME))
else:
    model.load_state_dict(torch.load('DIV/logs_srdn_rdn/model/' + MODEL_NAME, map_location=lambda storage, loc: storage))

test_image_path = os.path.join(os.getcwd(), IMAGE_SET)
test_image_name = os.listdir(test_image_path)

PSNR_col = []
SSIM_col = []
brisque_col = []
bicubic_PSNR_col = []
bicubic_SSIM_col = []
bicubic_brisque_col=[]


for test_image_id in range(len(test_image_name)):
    image_path = os.path.join(test_image_path, test_image_name[test_image_id])
    #image = cv2.imread(image_path)
    image = pil_image.open(image_path).convert('RGB')
#    image = np.array(image)
#    if image.shape[0]>400 and image.shape[1]>400:
#        image = image[0:400,0:400,:]
#    elif image.shape[0]>400:
#        image = image[0:400,:,:]
#    else:
#        image = image[:,0:400,:]
#    image = pil_image.fromarray(image).convert('RGB')
    image_input, image_label = bicubic_downsample(image, scale=opt.upscale_factor)
#    image_label = modcrop(image, scale=3)
    image_input = np.transpose(image_input.astype('float32'), [2,0,1])
    image_label = np.transpose(image_label.astype('float32'), [2,0,1])
    image_input = image_input/255.0
    image_label = image_label/255.0
    image_label = np.expand_dims(image_label, axis=0)
    image_input = np.expand_dims(image_input, axis=0)
    #print(image_label.shape)
    #print(image_input.shape)
    ISource = torch.Tensor(image_label)
    INoisy = torch.Tensor(image_input)
    noise = torch.FloatTensor(INoisy.size()).normal_(mean=0., std=TEST_NOISE_LEVEL/255.)
    INoisy = INoisy + noise
#    INoisy = INoisy
    #ISource, INoisy= Variable(ISource), Variable(INoisy)
    if TEST_MODE:
        #ISource = ISource.cuda()
        INoisy = INoisy.cuda()
    start = time.clock()
    out = model(INoisy)
    elapsed = (time.clock() - start)
    print('cost' + str(elapsed) + 's')
    out_img = out[0].data.detach().cpu().numpy()
    label_img = ISource[0].data.detach().cpu().numpy()
    noisy_img = INoisy[0].data.detach().cpu().numpy()
    out_img = np.transpose(out_img, [1,2,0])
    label_img = np.transpose(label_img, [1,2,0])
    noisy_img = np.transpose(noisy_img, [1,2,0])
    noisy_img = np.clip(noisy_img*255, 0, 255)
    noisy_img = pil_image.fromarray(noisy_img.astype('uint8')).convert('RGB')
    hr_bicubic = bicubic_sample(noisy_img, scale=opt.upscale_factor, denoise=True)
    hr_bicubic = hr_bicubic.astype('float32')/255.0
    hr_bicubic_gpu = np.transpose(hr_bicubic, [2,0,1])
    hr_bicubic_gpu = np.expand_dims(hr_bicubic_gpu, axis=0)
    hr_bicubic_gpu = torch.Tensor(hr_bicubic_gpu)
    
    #calculate PSNR
#    PSNR_image = compare_psnr(out_img, label_img, data_range=1)
    PSNR_image, SSIM_image = batch_PSNR(out, ISource, scale=3, data_range=255.0)
    PSNR_image_bicubic, SSIM_image_bicubic = batch_PSNR(hr_bicubic_gpu, ISource, scale=3, data_range=255.0)
    out_img = np.array(np.clip(out_img,0,1)*255, dtype='uint8')
    hr_bicubic = np.array(np.clip(hr_bicubic,0,1)*255, dtype='uint8')
    brisque_image = cv2.quality.QualityBRISQUE_compute(out_img, 'brisque_model_live.yml','brisque_range_live.yml')[0]
    brisque_image_bicubic = cv2.quality.QualityBRISQUE_compute(hr_bicubic, 'brisque_model_live.yml','brisque_range_live.yml')[0]
    PSNR_col.append(PSNR_image)
#    SSIM_image = compare_ssim(out_img, label_img, multichannel=True)
    SSIM_col.append(SSIM_image)
    brisque_col.append(brisque_image)
    bicubic_PSNR_col.append(PSNR_image_bicubic)
    bicubic_SSIM_col.append(SSIM_image_bicubic)
    bicubic_brisque_col.append(brisque_image_bicubic)
#    image_input = np.squeeze(image)
#    image_input = np.transpose(image,[1,2,0])
#    imsave(os.path.join(SAVE_PATH, test_image_name[test_image_id]), np.transpose(INoisy[0].data.detach().cpu().numpy(),[1,2,0]))
#    imsave(os.path.join(SAVE_PATH, test_image_name[test_image_id]), out_img)
    imsave(os.path.join(SAVE_PATH, test_image_name[test_image_id]), hr_bicubic)
    print('id: [%4d], psnr:[%.4f], ssim：[%.4f], brisque:[%.4f]'%(test_image_id, PSNR_image, SSIM_image, brisque_image))
    print('id: [%4d], psnr:[%.4f], ssim：[%.4f], brisque:[%.4f]'%(test_image_id, PSNR_image_bicubic, SSIM_image_bicubic, brisque_image_bicubic))

PSNR_avg = float(sum(PSNR_col))/len(PSNR_col)
SSIM_avg = float(sum(SSIM_col))/len(SSIM_col)
BRISQUE_avg = float(sum(brisque_col))/len(brisque_col)

print('average psnr:[%4f]'% PSNR_avg)
print('average ssim:[%4f]'%SSIM_avg)
print('average brisque:[%4f]'%BRISQUE_avg)

bicubic_PSNR_avg = float(sum(bicubic_PSNR_col))/len(bicubic_PSNR_col)
bicubic_SSIM_avg = float(sum(bicubic_SSIM_col))/len(bicubic_SSIM_col)
bicubic_BRISQUE_avg = float(sum(bicubic_brisque_col))/len(bicubic_brisque_col)

print('average psnr:[%4f]'% bicubic_PSNR_avg)
print('average ssim:[%4f]'% bicubic_SSIM_avg)
print('average brisque:[%4f]'% bicubic_BRISQUE_avg)
