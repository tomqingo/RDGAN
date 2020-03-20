# -*- coding: utf-8 -*-
"""
Created on Tue Dec 24 10:57:47 2019

@author: Knxie
"""

import os
import os.path
import numpy as np
import random
import h5py
import torch
import cv2
import glob
import torch.utils.data as udata
from my_utils import data_augmentation
import PIL.Image as pil_image
from PIL import ImageFilter

def normalize(data):
    return data/255.0

def modcrop(img, scale =3):
    if len(img.shape) ==3:
        h, w, _ = img.shape
        h = int((h / scale)) * scale
        w = int((w / scale)) * scale
        img = img[0:h, 0:w, :]
    else:
        h, w = img.shape
        h = int((h / scale)) * scale
        w = int((w / scale)) * scale
        img = img[0:h, 0:w]
    return img

def Im2Patch(img, win, stride=1):
    k = 0
    endc = img.shape[0]
    endw = img.shape[1]
    endh = img.shape[2]
    patch = img[:, 0:endw-win+0+1:stride, 0:endh-win+0+1:stride]
    TotalPatNum = patch.shape[1] * patch.shape[2]
    Y = np.zeros([endc, win*win,TotalPatNum], np.float32)
    for i in range(win):
        for j in range(win):
            patch = img[:,i:endw-win+i+1:stride,j:endh-win+j+1:stride]
            Y[:,k,:] = np.array(patch[:]).reshape(endc, TotalPatNum)
            k = k + 1
    return Y.reshape([endc, win, win, TotalPatNum])

def bicubic_downsample(img, scale, down_type=pil_image.BICUBIC):
    #out = cv2.resize(img, (int(img.shape[1]/scale), int(img.shape[0]/scale)), interpolation=down_type)
    hr = img.resize(((img.width//scale)*scale, (img.height//scale)*scale), resample=pil_image.BICUBIC)
    lr = img.resize((hr.width//scale, hr.height//scale), resample=pil_image.BICUBIC)
    hr = np.array(hr)
    lr = np.array(lr)
    return lr, hr

def bicubic_sample(img, scale, denoise=False):
    hr_upsample = img.resize((img.width*scale, img.height*scale), resample=pil_image.BICUBIC)
    if denoise:
        hr_upsample = hr_upsample.filter(ImageFilter.BLUR)
    hr_upsample = np.array(hr_upsample)
    return hr_upsample

def make_sub_data(path, data_org, label, train_num, h, w, scale, c):
    if train_num == 0:
        if os.path.exists(path):
            return False
        h5f = h5py.File(path, 'w')
        input_h5 = h5f.create_dataset('input', (1, c, int(h/scale), int(w/scale)),maxshape=(None, c, int(h/scale), int(w/scale)), 
                                      chunks=(1, c, int(h/scale), int(w/scale)), dtype='float32')
        label_h5 = h5f.create_dataset('label', (1, c, int(h/scale)*scale, int(w/scale)*scale),maxshape=(None, c, int(h/scale)*scale,int(w/scale)*scale), 
                                      chunks=(1, c, int(h/scale)*scale, int(w/scale)*scale), dtype='float32')
        input_h5.resize([train_num + 1, c, int(h/scale), int(w/scale)])
        input_h5[train_num:train_num+1] = data_org
        label_h5.resize([train_num + 1, c, int(h/scale)*scale, int(w/scale)*scale])
        label_h5[train_num:train_num+1] = label
    else:
        h5f = h5py.File(path, 'a')
        input_h5 = h5f['input']
        label_h5 = h5f['label']

        input_h5.resize([train_num + 1, c, int(h/scale), int(w/scale)])
        input_h5[train_num:train_num+1] = data_org
        label_h5.resize([train_num + 1, c, int(h/scale)*scale, int(w/scale)*scale])
        label_h5[train_num:train_num+1] = label
    
    h5f.close()
    return True


def prepare_data(data_path, patch_size, stride, scale_new):
    # train
    print('process training data')
    scale = int(scale_new[0])
#    scales = [1, 0.9, 0.8, 0.7]
    files = glob.glob(os.path.join(data_path, 'BSD500train', '*.jpg'))
    files.sort()
    train_num = 0
    for i in range(len(files)):
        #img = cv2.imread(files[i])
        img = pil_image.open(files[i]).convert('RGB')
        img_org = np.array(img)
        h, w, c = img_org.shape
#        for k in range(len(scales)):
        #Img = cv2.resize(img, (int(w*scales[k]), int(h*scales[k])), interpolation=cv2.INTER_CUBIC)
        #Img = img.transpose((2,0,1))
            #Img = np.expand_dims(Img[:,:,0].copy(), 0)
        Img_downsample, Img = bicubic_downsample(img, scale)
        #Img = normalize(Img)
        #Img = modcrop(img, scale=3)
        Img = Img.transpose((2,0,1))
        Img_downsample = Img_downsample.transpose((2,0,1))
        Img = Img.astype('float32')
        Img_downsample = Img_downsample.astype('float32')
        #Img_downsample = bicubic_downsample(Img, scale, cv2.INTER_CUBIC)
        patches_label = Im2Patch(Img, win=int(patch_size/scale)*scale, stride=int(stride/scale)*scale)
        patches_input = Im2Patch(Img_downsample, win=int(patch_size/scale), stride=int(stride/scale))
        print(patches_label.shape[3])
        print(patches_input.shape[3])
        for n in range(patches_input.shape[3]):
            data_org = patches_input[:,:,:,n].copy()
            label = patches_label[:,:,:,n].copy()
            #t = cv2.cvtColor(label.transpose(1,2,0), cv2.COLOR_BGR2YCR_CB)
            #t = t[:,:, 0]
            #gx = t[1:, 0:-1]-t[0:-1,0:-1]
            #gy = t[0:-1, 1:]-t[0:-1,0:-1]
            #Gxy = (gx**2+gy**2)**0.5
            #r_gxy = (float(((Gxy>10).sum()))/(int((patch_size/scale)*scale)**2))*100
            #if r_gxy < 10:
            #    continue
            #label = modcrop(data.transpose((1,2,0)), scale)
            #data_org = bicubic_downsample(label, scale, cv2.INTER_CUBIC)
            #label = label.transpose((2,0,1))
            #data_org = data_org.transpose((2,0,1))
            choice = np.random.randint(1,8)
            data_org = normalize(data_org)
            label = normalize(label)
            data_aug = data_augmentation(data_org, choice)
            label_aug = data_augmentation(label, choice)
                #label = modcrop(data_aug.transpose((1,2,0)), scale)
                #data_org = bicubic_downsample(label, scale, cv2.INTER_CUBIC)
                #label = label.transpose((2,0,1))
                #data_org = data_org.transpose((2,0,1))
                    #h5f.create_dataset(str(train_num)+"_aug_%d" % (m+1), data=data_aug)
            flag_train = make_sub_data('train_BSD500.h5', data_aug, label_aug, train_num, patch_size, patch_size, scale, c)
            train_num += 1
            if not flag_train:
                break

    # val
    print('\nprocess validation data')
    files.clear()
    files = glob.glob(os.path.join(data_path, 'BSD500test', '*.jpg'))
    files.sort()
    val_num = 0
    val_patch = 321
    for i in range(len(files)):
        print("file: %s" % files[i])
        #img = cv2.imread(files[i])
        img = pil_image.open(files[i]).convert('RGB')
        img = np.array(img)
        img = img[0:val_patch, 0:val_patch, :]
        img = pil_image.fromarray(img)
        data_org, label = bicubic_downsample(img, scale)
        #img = img.transpose((2,0,1))
        #img = np.float32(normalize(img))
        #label = modcrop(img.transpose((1,2,0)), scale)
        #data_org = bicubic_downsample(label, scale, cv2.INTER_CUBIC)
        #label = label.transpose((2,0,1))
        #data_org = data_org.transpose((2,0,1))
        data_org = normalize(data_org.astype('float32'))
        label = normalize(label.astype('float32'))
        data_org = data_org.transpose(2,0,1)
        label = label.transpose(2,0,1)
        flag_val = make_sub_data('val_BSD500.h5', data_org, label, val_num, val_patch, val_patch, scale, 3)
        val_num += 1
        if not flag_val:
            break
    print('training set, # samples %d\n' % train_num)
    print('val set, # samples %d\n' % val_num)

class TrainDataset(udata.Dataset):
    def __init__(self, num_group, batch_size, train=True):
        super(TrainDataset, self).__init__()
        self.train = train
        h5f = h5py.File('train_BSD500.h5', 'r')

        self.data_length = num_group*batch_size        
        self.input_ = h5f['input']
        self.label_ = h5f['label']
        self.input_ = np.array(self.input_)
        self.label_ = np.array(self.label_)
        self.batch_size = batch_size
        self.data_index = np.arange(self.input_.shape[0])
#        random.shuffle(self.keys)
        h5f.close()
#        self.keys = list(h5f.keys())
    def __len__(self):
        return self.data_length
    def __getitem__(self, index):
#        if self.train:
#            h5f = h5py.File('train_color.h5', 'r')
#        else:
#            h5f = h5py.File('val_color.h5', 'r')
#        key = self.keys[index]
#        data = np.array(h5f[key])
#        h5f.close()
        if index%self.batch_size == 0:
            np.random.shuffle(self.data_index)
            #print(self.data_index)
        idx = self.data_index[index%self.batch_size]
        img_org = self.input_[idx]
        img_label = self.label_[idx]
            
#        h5f.close()
        return torch.Tensor(img_org), torch.Tensor(img_label)

class ValDataset(udata.Dataset):
    def __init__(self, train=False):
        super(ValDataset, self).__init__()
        self.train = train
        h5f = h5py.File('val_BSD500.h5', 'r')        
        self.input_ = h5f['input']
        self.label_ = h5f['label']
        self.input_ = np.array(self.input_)
        self.label_ = np.array(self.label_)
        self.data_length = self.input_.shape[0]       
#        random.shuffle(self.keys)
        h5f.close()
#        self.keys = list(h5f.keys())
    def __len__(self):
        return self.data_length
    def __getitem__(self, index):
#        if self.train:
#            h5f = h5py.File('train_color.h5', 'r')
#        else:
#            h5f = h5py.File('val_color.h5', 'r')
#        key = self.keys[index]
#        data = np.array(h5f[key])
#        h5f.close()
        img_org = self.input_[index]
        img_label = self.label_[index]
            
#        h5f.close()
        return torch.Tensor(img_org), torch.Tensor(img_label)
