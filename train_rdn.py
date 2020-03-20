import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.utils as utils
from torch.autograd import Variable
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from model_rdn import make_model, Discriminator
from loss import GeneratorLoss
#from dataset import prepare_data, TrainDataset, ValDataset
from datasets import TrainDataset, EvalDataset
from skimage.io import imsave
from my_utils import *
import sys

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

parser = argparse.ArgumentParser(description="SRGAN-RDN")
parser.add_argument("--preprocess", type=bool, default=False, help='run prepare_data or not')
parser.add_argument('--upscale_factor', type=str, default='3', help='super resolution scale')
parser.add_argument('--patch_size', type=int, default=32, help='output patch size')
parser.add_argument('--G0', type=int, default=64, help='default number of filters. (Use in RDN)')
parser.add_argument('--RDNkSize', type=int, default=3, help='default kernel size. (Use in RDN)')
parser.add_argument('--RDNconfig', type=str, default='B', help='parameters config of RDN. (Use in RDN)')
parser.add_argument("--batchSize", type=int, default=16, help="Training batch size")
parser.add_argument("--num_of_layers", type=int, default=17, help="Number of total layers")
parser.add_argument("--epochs", type=int, default=150, help="Number of training epochs")
parser.add_argument("--milestone", type=int, default=30, help="When to decay learning rate; should be less than epochs")
parser.add_argument('--generatorLR', type=float, default=0.0001, help='learning rate for generator')
parser.add_argument('--discriminatorLR', type=float, default=0.0001, help='learning rate for discriminator')
parser.add_argument("--outf", type=str, default="logs_test", help='path of log files')
parser.add_argument("--mode", type=str, default="S", help='with known noise level (S) or blind training (B)')
parser.add_argument("--noiseL", type=float, default=25, help='noise level; ignored when mode=B')
parser.add_argument("--val_noiseL", type=float, default=25, help='noise level used on validation set')
parser.add_argument('--n_colors', type=int, default=3, help='number of color channels to use')
parser.add_argument('--cuda', type=bool, default=True, help='enable cuda')
opt = parser.parse_args()

def save_result(result,path):
    path = path if path.find('.') != -1 else path+'.png'
    imsave(path, np.clip(result,0,1))

def make_grid(images,nrow,nline):
    rows=[]
    for i in range(nrow):
        tmp=images[i]
        for j in range(nline-1):
            tmp=np.vstack((tmp,images[(j+1)*nrow+i]))
        rows.append(tmp)
    tmp=rows[0]
    for i in range(nrow-1):
        tmp=np.hstack((tmp,rows[i+1]))
    return tmp

def main():
    # Load dataset
    print('Loading dataset ...\n')
    #dataset_train = TrainDataset(1000, opt.batchSize, train=True)
    #dataset_val = ValDataset(train=False)
    dataset_train = TrainDataset('train_BSD500.h5', opt.patch_size, int(opt.upscale_factor[0]))
    print(len(dataset_train))
    dataset_val = EvalDataset('test_BSD500.h5')
    loader_train = DataLoader(dataset=dataset_train, num_workers=1, batch_size=opt.batchSize, shuffle=True)
    loader_val = DataLoader(dataset=dataset_val, batch_size=1)
    print("# of training samples: %d\n" % int(len(dataset_train)))
    # Build model
    netG = make_model(opt)
    print('# generator parameters:', sum(param.numel() for param in netG.parameters()))
    #netG.apply(weights_init_kaiming)
#    content_criterion = nn.MSELoss()
    #feature_extractor = FeatureExtractor(torchvision.models.vgg19(pretrained=True))
    content_criterion = nn.L1Loss()
    
    # Move to GPU
    if torch.cuda.is_available():
        netG.cuda()
        content_criterion.cuda()
    
    optim_rdn = optim.Adam(netG.parameters(), lr=opt.generatorLR)
    # Optimizer
    # training
    writer = SummaryWriter(opt.outf)
    step = 0
#    noiseL_B=[0,55] # ingnored when opt.mode=='S'
    
    # Generator Pretraining(Using MSE Loss)
    for epoch in range(opt.epochs):
        mean_generator_content_loss = 0.0
        mean_generator_PSNRs = 0.0
        mean_generator_SSIMs = 0.0
        for param_group in optim_rdn.param_groups:
            param_group['lr'] = opt.generatorLR * (0.1 ** (epoch // int(opt.epochs * 0.8)))
        for i, (lrimg, hrimg) in enumerate(loader_train):
            # adding noise
            for j in range(opt.batchSize):
                #noise = torch.FloatTensor(lrimg[j].size()).normal_(mean=0.0, std=opt.noiseL/255.)
                #lrimg[j] = lrimg[j] + noise
                lrimg[j] = lrimg[j] 
            # Generate real and fake inputs
            if opt.cuda:
                high_res_real = Variable(hrimg.cuda())
                high_res_fake = netG(Variable(lrimg).cuda())
            else:
                high_res_real = Variable(hrimg)
                high_res_fake = netG(Variable(lrimg))
    
            ######### Train generator #########
            netG.zero_grad()
    
            generator_content_loss = content_criterion(high_res_fake, high_res_real)
            mean_generator_content_loss += generator_content_loss.data
    
            generator_content_loss.backward()
            optim_rdn.step()

                        ######### Status and display #########
            sys.stdout.write('\r[%d/%d][%d/%d] Generator_MSE_Loss: %.4f' % (epoch, opt.epochs, i, len(loader_train), generator_content_loss.data))
          #  visualizer.show(low_res, high_res_real.cpu().data, high_res_fake.cpu().data)
            out_train = torch.clamp(high_res_fake, 0., 1.)
            psnr_train, ssim_train = batch_PSNR(out_train, high_res_real, scale=3, data_range=255.0)
            mean_generator_PSNRs += psnr_train
            mean_generator_SSIMs += ssim_train

            if step % 10 == 0:
                # Log the scalar values
                writer.add_scalar('generator_content_loss', generator_content_loss.item(), step)
                #writer.add_scalar('PSNR on training data', psnr_train, step)
                #writer.add_scalar('SSIM on training data', ssim_train, step)
            step += 1   
          #  sys.stdout.write('\r[%d/%d][%d/%d] PSNR: %.4f, SSIM:%.4f' % (epoch, 2, i, len(loader_train), psnr_train, ssim_train))
    
        psnr_avg_train = mean_generator_PSNRs/len(loader_train)
        ssim_avg_train = mean_generator_SSIMs/len(loader_train)
        sys.stdout.write('\r[%d/%d][%d/%d] Generator_MSE_Loss: %.4f\n' % (epoch, opt.epochs, i, len(loader_train), mean_generator_content_loss/len(loader_train)))
        print("\n[epoch %d] PSNR_train: %.4f" % (epoch+1, psnr_avg_train))
        print("\n[epoch %d] SSIM_train: %.4f" % (epoch+1, ssim_avg_train))
        writer.add_scalar('PSNR on training data', psnr_avg_train, epoch)
        writer.add_scalar('SSIM on training data', ssim_avg_train, epoch)
        #log_value('generator_mse_loss', mean_generator_content_loss/len(dataloader), epoch)
        torch.save(netG.state_dict(), '%s/model/rdn_final_%d.pth'%(opt.outf,epoch))

        ## the end of each epoch
        # netG.eval()
        # validate
        psnr_val = 0
        ssim_val = 0.0
        val_images = []
        num = 0
        numofex=opt.noiseL

        for index, (lrimg_val, hrimg_val) in enumerate(loader_val):
            #lrimg_val, hrimg_val = dataset_val[k]
            #noise = torch.FloatTensor(lrimg_val.size()).normal_(mean=0, std=opt.val_noiseL/255.)
            #lrimgn_val = lrimg_val + noise
            lrimgn_val = lrimg_val 
            #lrimgn_val = torch.Tensor(np.expand_dims(lrimgn_val, axis=0))
            #hrimg_val = torch.Tensor(np.expand_dims(hrimg_val, axis=0))
            hrimg_val, lrimg_val = Variable(hrimg_val.cuda(), volatile=True), Variable(lrimgn_val.cuda(), volatile=True)
            out_val = netG(lrimg_val)
            psnr_val_e, ssim_val_e = batch_PSNR(out_val, hrimg_val, scale=3, data_range=255.0)
            psnr_val += psnr_val_e
            ssim_val += ssim_val_e
            hrimg_val = np.transpose(hrimg_val[0].detach().cpu().numpy(), (1,2,0))
            out_val = np.transpose(out_val[0].detach().cpu().numpy(),(1,2,0))
            
            if num<5:
                num+=1
#                hrimg_val = hrimg_val[int(hrimg_val.shape[0] / 2) - 160:int(hrimg_val.shape[0] / 2) + 160,
#                    int(hrimg_val.shape[1] / 2) - 160:int(hrimg_val.shape[1] / 2) + 160]
#                out_val = out_val[int(out_val.shape[0] / 2) - 160:int(out_val.shape[0] / 2) + 160,
#                    int(out_val.shape[1] / 2) - 160:int(out_val.shape[1] / 2) + 160]
                val_images.extend([hrimg_val,out_val])


        output_image=make_grid(val_images,nrow=2,nline=1)
        if not os.path.exists('%s/training_results/%d/' % (opt.outf, numofex)):
            os.makedirs('%s/training_results/%d/' % (opt.outf, numofex))
        save_result(output_image,path='%s/training_results/%d/epoch%d.png' % (opt.outf,numofex,epoch))

        psnr_val /= len(dataset_val)
        ssim_val /= len(dataset_val)
        print("\n[epoch %d] PSNR_val: %.4f" % (epoch+1, psnr_val))
        print("\n[epoch %d] SSIM_val: %.4f" % (epoch+1, ssim_val))
        writer.add_scalar('PSNR on validation data', psnr_val, epoch)
        writer.add_scalar('SSIM on validation data', ssim_val, epoch)


if __name__ == "__main__":
    if opt.preprocess:
        prepare_data(data_path='data', patch_size=opt.patch_size, stride=30, scale_new=opt.upscale_factor)
    main()
