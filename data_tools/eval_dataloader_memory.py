#!/usr/bin/env python3
# -*- coding:utf-8 -*-
#############################################################
# File: eval_dataloader_memory.py
# Created Date: Thursday March 11th 2021
# Author: Chen Xuanhong
# Email: chenxuanhongzju@outlook.com
# Last Modified:  Thursday, 20th April 2023 9:27:26 am
# Modified By: Chen Xuanhong
# Copyright (c) 2021 Shanghai Jiao Tong University
#############################################################


import os
import cv2
import glob
import torch
from tqdm import tqdm

class EvalDataset:
    def __init__(   self,
                    dataset_name,
                    data_root,
                    batch_size  = 1,
                    degradation = "bicubic",
                    image_scale = 4,
                    subffix='png'):
        """Initialize and preprocess the urban100 dataset."""
        self.data_root      = data_root
        self.degradation    = degradation
        self.image_scale    = image_scale
        self.dataset_name   = dataset_name
        self.subffix        = subffix
        self.dataset        = []
        self.pointer        = 0
        self.batch_size     = 1

        if self.dataset_name.lower() == "set5":
            self.dataset_name = "Set5"
        elif self.dataset_name.lower() == "set14":
            self.dataset_name = "Set14"
        elif self.dataset_name.lower() == "b100":
            self.dataset_name = "B100"
        elif self.dataset_name.lower() == "urban100":
            self.dataset_name = "Urban100"
        # elif self.dataset_name.lower() == "div2k":
        #     self.dataset_name = "div2k"
        else:
            raise FileNotFoundError
        print("%s dataset is used!"%self.dataset_name)

        self.__preprocess__()
        self.num_images     = len(self.dataset)

        
        # c_transforms  = []
        # c_transforms.append(T.ToTensor())
        # c_transforms.append(T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)))
        # self.img_transform = T.Compose(c_transforms)
    
    def __preprocess__(self):
        """Preprocess the Urban100 dataset."""
        set5hr_path  = os.path.join(self.data_root, "HR", self.dataset_name, "x%d"%self.image_scale)
        set5lr_path  = os.path.join(self.data_root, "LR", "LRBI", self.dataset_name,"x%d"%self.image_scale)
        print("Evaluation dataset HR path: %s"%set5hr_path)
        print("Evaluation dataset LR path: %s"%set5lr_path)
        assert os.path.exists(set5hr_path)
        assert os.path.exists(set5lr_path)
        data_paths  = []    
        print("processing %s images..."%self.dataset_name)
        temp_path   = os.path.join(set5hr_path,'*.%s'%(self.subffix))
        images      = glob.glob(temp_path)
        print(images)
        for item in images:
            file_name   = os.path.basename(item).replace('HR' , 'LRBI')
            lr_name     = os.path.join(set5lr_path, file_name)
            data_paths.append([item,lr_name])
        
        for item_pair in tqdm(data_paths):
            hr_img      = cv2.imread(item_pair[0])
            hr_img      = cv2.cvtColor(hr_img,cv2.COLOR_BGR2RGB)
            hr_img      = hr_img.transpose((2,0,1))
            hr_img      = torch.from_numpy(hr_img)
            
            lr_img      = cv2.imread(item_pair[1])
            lr_img      = cv2.cvtColor(lr_img,cv2.COLOR_BGR2RGB)
            lr_img      = lr_img.transpose((2,0,1))
            lr_img      = torch.from_numpy(lr_img)
            
            self.dataset.append((hr_img,lr_img))
        # self.dataset = images
        print('Finished preprocessing the Urban100 Validation dataset, total image number: %d...'%len(self.dataset))

    def __call__(self):
        """Return one batch images."""
        if self.pointer>=self.num_images:
            self.pointer = 0
        hr = self.dataset[self.pointer][0]
        lr = self.dataset[self.pointer][1]
        hr = (hr/255.0 - 0.5) * 2.0
        lr = (lr/255.0 - 0.5) * 2.0
        hr = hr.unsqueeze(0)
        lr = lr.unsqueeze(0)
        self.pointer += 1
        return hr, lr
    
    def __len__(self):
        return self.num_images

    def __repr__(self):
        return self.__class__.__name__ + ' (' + self.data_root + ')'