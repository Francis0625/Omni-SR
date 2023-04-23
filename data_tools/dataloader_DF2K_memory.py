#!/usr/bin/env python3
# -*- coding:utf-8 -*-
#############################################################
# File: dataloader_DF2K_memory.py
# Created Date: Saturday April 4th 2020
# Author: Chen Xuanhong
# Email: chenxuanhongzju@outlook.com
# Last Modified:  Sunday, 23rd April 2023 3:10:56 pm
# Modified By: Chen Xuanhong
# Copyright (c) 2020 Shanghai Jiao Tong University
#############################################################

import os
import cv2
import glob
import torch
import random
import numpy as np
from tqdm import tqdm
from torch.utils import data
from torchvision import transforms as T



class DataPrefetcher():
    def __init__(self, loader):
        self.loader = loader
        self.dataiter = iter(loader)
        self.stream = torch.cuda.Stream()
        # self.mean = torch.tensor([0.485 * 255, 0.456 * 255, 0.406 * 255]).cuda().view(1,3,1,1)
        # self.std = torch.tensor([0.229 * 255, 0.224 * 255, 0.225 * 255]).cuda().view(1,3,1,1)
        # With Amp, it isn't necessary to manually convert data to half.
        # if args.fp16:
        #     self.mean = self.mean.half()
        #     self.std = self.std.half()
        self.__preload__()

    def __preload__(self):
        try:
            self.hr, self.lr = next(self.dataiter)
        except StopIteration:
            self.dataiter = iter(self.loader)
            self.hr, self.lr = next(self.dataiter)
            
        with torch.cuda.stream(self.stream):
            self.hr     = self.hr.cuda(non_blocking=True)
            self.lr     = self.lr.cuda(non_blocking=True)
            self.hr     = (self.hr/255.0 - 0.5) * 2.0
            self.lr     = (self.lr/255.0 - 0.5) * 2.0
            # With Amp, it isn't necessary to manually convert data to half.
            # if args.fp16:
            #     self.next_input = self.next_input.half()
            # else:
            # self.next_input = self.next_input.float()
            # self.next_input = self.next_input.sub_(self.mean).div_(self.std)

    def next(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        hr  = self.hr
        lr  = self.lr
        self.__preload__()
        return hr, lr
    
    def __len__(self):
        """Return the number of images."""
        return len(self.loader)

class DIV2K_Flickr_Dataset(data.Dataset):
    """Dataset class for the Artworks dataset and content dataset."""
    def __init__(   self,
                    flickr_root,
                    div2k_root,
                    img_transform,
                    lr_patch_size,
                    degradation = "bicubic",
                    image_scale = 4,
                    subffix='png',
                    random_seed=1234,
                    dataset_enlarge=50,
                    multitask_num=8):
        """Initialize and preprocess the flickr and div2k dataset."""
        self.flickr_root    = flickr_root
        self.div2k_root     = div2k_root
        # self.flickr_root    = '/Home/data/Flickr2K'
        # self.div2k_root     = '/Home/data/DIV2K'
        self.degradation    = degradation
        self.i_s            = image_scale
        self.l_ps           = lr_patch_size
        self.h_ps           = lr_patch_size* image_scale
        self.d_e            = dataset_enlarge
        self.img_transform  = img_transform
        self.subffix        = subffix
        self.task_num       = multitask_num
        self.dataset        = []
        self.random_seed    = random_seed
        random.seed(self.random_seed)
        self.__preprocess__()
        self.num_images = len(self.pathes)

    def __preprocess__(self):
        """Preprocess the Artworks dataset."""
        
        data_path_list = []

        flickrhr_path = os.path.join(self.flickr_root,"Flickr2K_HR")#.replace('/','_'))
        div2khr_path  = os.path.join(self.div2k_root,"DIV2K_train_HR")#.replace('/','_'))
        if self.degradation == "bicubic":    
            flickrlr_path = os.path.join(self.flickr_root,"Flickr2K_LR_bicubic", "X%d"%self.i_s)
            div2klr_path  = os.path.join(self.div2k_root,"DIV2K_train_LR_bicubic", "X%d"%self.i_s)
        else:
            flickrlr_path = os.path.join(self.flickr_root,"Flickr2K_LR_unknown", "X%d"%self.i_s)
            div2klr_path  = os.path.join(self.div2k_root,"DIV2K_train_LR_unknown", "X%d"%self.i_s)
        
        print("processing Flickr images...")
        temp_path   = os.path.join(flickrhr_path,'*.%s'%(self.subffix))
        images      = glob.glob(temp_path)
        for item in images:
        # for item in images[0:100]:
            file_name   = os.path.basename(item)
            file_name   = os.path.splitext(file_name)[0]
            lr_name     = file_name+"x%d.%s"%(self.i_s,self.subffix)
            lr_name     = os.path.join(flickrlr_path,lr_name)
            data_path_list.append([item,lr_name])


        print("processing DIV2K images...")
        temp_path   = os.path.join(div2khr_path,'*.%s'%(self.subffix))
        images      = glob.glob(temp_path)
        for item in images:
        # for item in images[0:100]:
            file_name   = os.path.basename(item)
            file_name   = os.path.splitext(file_name)[0]
            lr_name     = file_name+"x%d.%s"%(self.i_s,self.subffix)
            lr_name     = os.path.join(div2klr_path,lr_name)
            data_path_list.append([item,lr_name])
        
        random.shuffle(data_path_list)
        # self.dataset = images
        print('Finished preprocessing the Flickr and DIV2K dataset, total image number: %d...'%len(data_path_list))

        for item_pair in tqdm(data_path_list[:]):
            hr_img      = cv2.imread(item_pair[0])
            hr_img      = cv2.cvtColor(hr_img,cv2.COLOR_BGR2RGB)
            hr_img      = hr_img.transpose((2,0,1))
            hr_img      = torch.from_numpy(hr_img)
            
            lr_img      = cv2.imread(item_pair[1])
            lr_img      = cv2.cvtColor(lr_img,cv2.COLOR_BGR2RGB)
            lr_img      = lr_img.transpose((2,0,1))
            lr_img      = torch.from_numpy(lr_img)
            
            self.dataset.append((hr_img,lr_img))
        indeices = np.random.randint(0,len(self.dataset),size=self.d_e*len(self.dataset))
        self.pathes= indeices.tolist()
        print("Finish to read the dataset!")
 

    def __getitem__(self, index):
        """Return one hr image and its corresponding lr image."""
        hr_img  = self.dataset[self.pathes[index]][0]
        lr_img  = self.dataset[self.pathes[index]][1]

        hight   = lr_img.shape[1] # h
        width   = lr_img.shape[2]

        r_h     = random.randint(0,hight-self.l_ps)
        r_w     = random.randint(0,width-self.l_ps)
        
        hr_img  = hr_img[:,r_h * self.i_s:(r_h * self.i_s + self.h_ps),
                                r_w * self.i_s:(r_w * self.i_s + self.h_ps)]
        lr_img  = lr_img[:,r_h:(r_h+self.l_ps),r_w:(r_w+self.l_ps)]

        
        flip_ran= random.randint(0,2)
        
        if flip_ran == 0:
            # horizontal
            hr_img  = torch.flip(hr_img,[1])
            lr_img  = torch.flip(lr_img,[1])
        elif flip_ran == 1:
            # vertical
            hr_img  = torch.flip(hr_img,[2])
            lr_img  = torch.flip(lr_img,[2])
        
        rot_ran = random.randint(0,3)

        if rot_ran != 0:
            # horizontal
            hr_img  = torch.rot90(hr_img, rot_ran, [1, 2])
            lr_img  = torch.rot90(lr_img, rot_ran, [1, 2])



        # hr_img  = (hr_img/255.0 - 0.5) * 2.0
        # hr_img  = hr_img.float()
        

        # lr_img  = (lr_img/255.0 - 0.5) * 2.0
        # lr_img  = lr_img.float()
        # lr_img  = torch.from_numpy(lr_img)
        
        return hr_img,lr_img

    def __len__(self):
        """Return the number of images."""
        return self.num_images

def GetLoader(  dataset_roots,
                batch_size=16,
                random_seed=1234,
                **kwargs
                ):
    """Build and return a data loader."""
    if not kwargs:
        a = "Input params error!"
        raise ValueError(print(a))
    colorJitterEnable = kwargs["color_jitter"]
    colorConfig       = kwargs["color_config"]
    num_workers       = kwargs["dataloader_workers"]
    degradation       = kwargs["degradation"]
    image_scale       = kwargs["image_scale"]
    lr_patch_size     = kwargs["lr_patch_size"]
    subffix           = kwargs["subffix"]
    num_workers       = kwargs["dataloader_workers"]
    # flickr_root       = dataset_roots["Flickr"]
    # div2k_root        = dataset_roots["DIV2K"]
    flickr_root       = '/Home/data/Flickr2K'
    div2k_root         = '/Home/data/DIV2K'
    dataset_enlarge   = kwargs["dataset_enlarge"]
    
    c_transforms = []

    c_transforms.append(T.RandomHorizontalFlip())
    
    c_transforms.append(T.RandomVerticalFlip())

    if colorJitterEnable:
        if colorConfig is not None:
            print("Enable color jitter!")
            colorBrightness = colorConfig["brightness"]
            colorContrast   = colorConfig["contrast"]
            colorSaturation = colorConfig["saturation"]
            colorHue        = (-colorConfig["hue"],colorConfig["hue"])
            c_transforms.append(T.ColorJitter(brightness=colorBrightness,\
                                contrast=colorContrast,saturation=colorSaturation, hue=colorHue))
    c_transforms.append(T.ToTensor())
    c_transforms.append(T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)))

    c_transforms = T.Compose(c_transforms)

    content_dataset = DIV2K_Flickr_Dataset(
                        flickr_root,
                        div2k_root,
                        c_transforms,
                        lr_patch_size,
                        degradation,
                        image_scale,
                        subffix,
                        random_seed,
                        dataset_enlarge)

    content_data_loader = data.DataLoader(
                        dataset=content_dataset,
                        batch_size=batch_size,
                        drop_last=True,
                        shuffle=True,
                        num_workers=num_workers,
                        pin_memory=True)
                        
    prefetcher = DataPrefetcher(content_data_loader)
    return prefetcher
