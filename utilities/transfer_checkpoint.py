#!/usr/bin/env python3
# -*- coding:utf-8 -*-
#############################################################
# File: transfer_checkpoint.py
# Created Date: Wednesday February 3rd 2021
# Author: Chen Xuanhong
# Email: chenxuanhongzju@outlook.com
# Last Modified:  Thursday, 4th February 2021 1:27:09 pm
# Modified By: Chen Xuanhong
# Copyright (c) 2021 Shanghai Jiao Tong University
#############################################################
import torch
from torch import nn as nn
from torch.nn import functional as F
from torch.nn import init as init
import os
import numpy as np
import scipy.io as io

class RepSRPlain_pixel(nn.Module):
    """Networks consisting of Residual in Residual Dense Block, which is used
    in ESRGAN.
    ESRGAN: Enhanced Super-Resolution Generative Adversarial Networks.
    Currently, it supports x4 upsampling scale factor.
    Args:
        num_in_ch (int): Channel number of inputs.
        num_out_ch (int): Channel number of outputs.
        num_feat (int): Channel number of intermediate features.
            Default: 64
        num_block (int): Block number in the trunk network. Defaults: 23
        num_grow_ch (int): Channels for each growth. Default: 32.
    """

    def __init__(self,
                 num_in_ch,
                 num_out_ch,
                 num_feat=32,
                 num_layer = 3,
                 upsampling=4):
        super(RepSRPlain_pixel, self).__init__()

        self.scale      = upsampling
        self.ssqu       = upsampling ** 2
        
        self.rep1      = nn.Conv2d(num_in_ch, num_feat,3,1,1)
        self.rep2      = nn.Conv2d(num_feat, num_feat*2,3,1,1)
        self.rep3      = nn.Conv2d(num_feat*2, num_feat*2,3,1,1)
        self.rep4      = nn.Conv2d(num_feat*2, num_feat*2,3,1,1)
        self.rep5      = nn.Conv2d(num_feat*2, num_feat*2,3,1,1)
        self.rep6      = nn.Conv2d(num_feat*2, num_feat,3,1,1)

        self.conv_up1   = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_up2   = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_hr    = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_last  = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)

        
        self.activator = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        # self.activator  = nn.ReLU(inplace=True)

        # default_init_weights([self.conv_up1,self.conv_up2,self.conv_hr,self.conv_last], 0.1)

    def forward(self, x):
        
        f_d  = self.activator(self.rep1(x))
        f_d  = self.activator(self.rep2(f_d))
        f_d  = self.activator(self.rep3(f_d))
        f_d  = self.activator(self.rep4(f_d))
        f_d  = self.activator(self.rep5(f_d))
        f_d  = self.activator(self.rep6(f_d))
        
        feat = self.activator(
            self.conv_up1(F.interpolate(f_d, scale_factor=2, mode='nearest')))
        feat = self.activator(
            self.conv_up2(F.interpolate(feat, scale_factor=2, mode='nearest')))
        out = self.conv_last(self.activator(self.conv_hr(feat)))
        return out

def create_identity_conv(dim,kernel_size=3):
    zeros = torch.zeros((dim,dim,kernel_size,kernel_size)).cuda()
    for i_dim in range(dim):
        zeros[i_dim,i_dim,kernel_size//2,kernel_size//2] = 1.0
    return zeros

def fill_conv_kernel(in_tensor,kernel_size=3):
    shape = in_tensor.shape
    zeros = torch.zeros(shape[0],shape[1],kernel_size,kernel_size).cuda()
    for i_dim in range(shape[0]):
        zeros[i_dim,:,kernel_size//2,kernel_size//2] = in_tensor[i_dim,:,0,0]
    return zeros

if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = str(0)
    base_path = "H:\\Multi Scale Kernel Prediction Networks\\Mobile_Oriented_KPN\\train_logs\\"
    version = "repsr_pixel_0"
    epoch   = 73
    path_ckp= os.path.join(base_path,version,"checkpoints\\epoch%d_RepSR.pth"%epoch)
    path_plain_ckp= os.path.join(base_path,version,"checkpoints\\epoch%d_RepSR_Plain.pth"%epoch)
    network = RepSRPlain_pixel(3,
                            3,
                            64,
                            3,
                            4
                            )
    network = network.cuda()

    
    
    wocao = network.state_dict()
    # for data_key in wocao.keys():
    #     print(data_key)
    #     print(wocao[data_key].shape)
    wocao_cpk = torch.load(path_ckp)
    
    # for data_key in wocao_cpk.keys():
    #     print(data_key)
    #     print(wocao_cpk[data_key].shape)
    name_list = ["rep1","rep2","rep3","rep4","rep5","rep6"]
    other_list = ["conv_up1","conv_up2","conv_hr","conv_last"]
    for i_name in name_list:
        temp= wocao_cpk[i_name+".conv3.weight"] + fill_conv_kernel(wocao_cpk[i_name+".conv1x1.weight"])
        wocao[i_name+".weight"] = temp
        temp= wocao_cpk[i_name+".conv3.bias"] + wocao_cpk[i_name+".conv1x1.bias"]
        wocao[i_name+".bias"] = temp

        if wocao_cpk[i_name+".conv3.weight"].shape[0] == wocao_cpk[i_name+".conv3.weight"].shape[1]:
            print("include identity")
            temp = wocao[i_name+".weight"] + create_identity_conv(wocao_cpk[i_name+".conv3.weight"].shape[0])
            wocao[i_name+".weight"] = temp
        
    for i_name in other_list:
        wocao[i_name+".weight"] = wocao_cpk[i_name+".weight"]
        wocao[i_name+".bias"]  =  wocao_cpk[i_name+".bias"]
        
    torch.save(wocao,path_plain_ckp)

    # wocao = torch.load(path_plain_ckp)
    # for data_key in wocao.keys():
    #     result1 = wocao[data_key].cpu().numpy()
    #     # np.savetxt(i_name+"_conv3_weight.txt",result1)
    #     str_temp = ("%s"%data_key).replace(".","_")
    #     io.savemat(str_temp+".mat",{str_temp:result1})
        
    # for data_key in wocao.keys():
    #     print(data_key)
    #     print(wocao[data_key].shape)