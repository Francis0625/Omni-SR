#!/usr/bin/env python3
# -*- coding:utf-8 -*-
#############################################################
# File: test_dataloader_rcan.py
# Created Date: Tuesday January 12th 2021
# Author: Chen Xuanhong
# Email: chenxuanhongzju@outlook.com
# Last Modified:  Thursday, 20th April 2023 9:31:19 am
# Modified By: Chen Xuanhong
# Copyright (c) 2021 Shanghai Jiao Tong University
#############################################################

import os
import glob
import torch
from PIL import Image
from pathlib import Path
from torchvision import transforms as T

class TestDataset:
    def __init__(   self,
                    dataset_name,
                    data_root,
                    batch_size  = 16,
                    degradation = "bicubic",
                    image_scale = 4,
                    subffix='png'):
        """Initialize and preprocess the B100 dataset."""
        self.data_root      = data_root
        self.image_scale    = image_scale
        self.dataset_name   = dataset_name
        self.subffix        = subffix
        self.dataset        = []
        self.pointer        = 0
        self.batch_size     = batch_size
        self.__preprocess__()
        self.num_images = len(self.dataset)

        if self.dataset_name.lower() == "set5":
            self.dataset_name = "Set5"
        elif self.dataset_name.lower() == "set14":
            self.dataset_name = "Set14"
        elif self.dataset_name.lower() == "b100":
            self.dataset_name = "B100"
        elif self.dataset_name.lower() == "urban100":
            self.dataset_name = "Urban100"

        c_transforms  = []
        c_transforms.append(T.ToTensor())
        c_transforms.append(T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)))
        self.img_transform = T.Compose(c_transforms)
    
    def __preprocess__(self):
        """Preprocess the B100 dataset."""
        
        set5hr_path  = os.path.join(self.data_root, "HR", self.dataset_name, "x%d"%self.image_scale)
        set5lr_path  = os.path.join(self.data_root, "LR", "LRBI", self.dataset_name,"x%d"%self.image_scale)
        
        print("processing %s images..."%self.dataset_name)
        # import pdb; pdb.set_trace()
        # temp_path   = os.path.join(set5hr_path,'*.%s'%(self.subffix))
        temp_path   = os.path.join(set5hr_path,'*.%s'%('png'))
        images      = glob.glob(temp_path)
        for item in images:
            file_name   = os.path.basename(item).replace('HR' , 'LRBI')
            lr_name     = os.path.join(set5lr_path, file_name)
            self.dataset.append([item,lr_name])
        # self.dataset = images
        print('Finished preprocessing the %s dataset, total image number: %d...'%(self.dataset_name,len(self.dataset)))

    def __call__(self):
        """Return one batch images."""
        if self.pointer>=self.num_images:
            self.pointer = 0
            a = "The end of the story!"
            raise StopIteration(print(a))
        filename= self.dataset[self.pointer][0]
        image   = Image.open(filename)
        hr      = self.img_transform(image)
        filename= self.dataset[self.pointer][1]
        image   = Image.open(filename)
        lr      = self.img_transform(image)
        file_name   = os.path.basename(filename)
        file_name   = os.path.splitext(file_name)[0]
        hr_ls   = hr.unsqueeze(0)
        lr_ls   = lr.unsqueeze(0)
        nm_ls   = [file_name,]

        self.pointer += 1
        return hr_ls, lr_ls, nm_ls
    
    def __len__(self):
        return self.num_images

    def __repr__(self):
        return self.__class__.__name__ + ' (' + self.data_root + ')'