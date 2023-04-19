#!/usr/bin/env python3
# -*- coding:utf-8 -*-
#############################################################
# File: data_loader.py
# Created Date: Saturday April 4th 2020
# Author: Chen Xuanhong
# Email: chenxuanhongzju@outlook.com
# Last Modified:  Tuesday, 5th January 2021 2:12:29 pm
# Modified By: Chen Xuanhong
# Copyright (c) 2020 Shanghai Jiao Tong University
#############################################################

import os
import glob
import torch
import random
from PIL import Image
from pathlib import Path
from torch.utils import data
import torchvision.datasets as dsets
from torchvision import transforms as T
import torchvision.transforms.functional as F


class StyleResize(object):
    def __call__(self, images):
        th, tw = images.size # target height, width
        if max(th,tw) > 1800:
            alpha = 1800. / float(min(th,tw))
            h     = int(th*alpha)
            w     = int(tw*alpha)
            images  = F.resize(images, (h, w))
        if max(th,tw) < 800:
            # Resize the smallest side of the image to 800px
            alpha = 800. / float(min(th,tw))
            if alpha < 4.:
                h     = int(th*alpha)
                w     = int(tw*alpha)
                images  = F.resize(images, (h, w))
            else:
                images  = F.resize(images, (800, 800))
        return images

    def __repr__(self):
        return self.__class__.__name__ + '()'

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
            self.content, self.style, self.label = next(self.dataiter)
        except StopIteration:
            self.dataiter = iter(self.loader)
            self.content, self.style, self.label = next(self.dataiter)
            
        with torch.cuda.stream(self.stream):
            self.content= self.content.cuda(non_blocking=True)
            self.style  = self.style.cuda(non_blocking=True)
            self.label  = self.label.cuda(non_blocking=True)
            # With Amp, it isn't necessary to manually convert data to half.
            # if args.fp16:
            #     self.next_input = self.next_input.half()
            # else:
            # self.next_input = self.next_input.float()
            # self.next_input = self.next_input.sub_(self.mean).div_(self.std)

    def next(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        content = self.content
        style   = self.style
        label   = self.label 
        self.__preload__()
        return content, style, label
    
    def __len__(self):
        """Return the number of images."""
        return len(self.loader)

class TotalDataset(data.Dataset):
    """Dataset class for the Artworks dataset and content dataset."""
    def __init__(self, content_image_dir,style_image_dir,
                    selectedContent,selectedStyle,
                    content_transform,style_transform,
                    subffix='jpg', random_seed=1234):
        """Initialize and preprocess the CelebA dataset."""
        self.content_image_dir= content_image_dir
        self.style_image_dir  = style_image_dir
        self.content_transform= content_transform
        self.style_transform  = style_transform
        self.selectedContent  = selectedContent
        self.selectedStyle    = selectedStyle
        self.subffix            = subffix
        self.content_dataset    = []
        self.art_dataset        = []
        self.random_seed= random_seed
        self.__preprocess__()
        self.num_images = len(self.content_dataset)
        self.art_num    = len(self.art_dataset)

    def __preprocess__(self):
        """Preprocess the Artworks dataset."""
        print("processing content images...")
        for dir_item in self.selectedContent:
            join_path = Path(self.content_image_dir,dir_item)#.replace('/','_'))
            if join_path.exists():
                print("processing %s"%dir_item)
                images = join_path.glob('*.%s'%(self.subffix))
                for item in images:
                    self.content_dataset.append(item)
            else:
                print("%s dir does not exist!"%dir_item)
        label_index = 0
        print("processing style images...")
        for class_item in self.selectedStyle:
            images = Path(self.style_image_dir).glob('%s/*.%s'%(class_item, self.subffix))
            for item in images:
                self.art_dataset.append([item, label_index])
            label_index += 1
        random.seed(self.random_seed)
        random.shuffle(self.content_dataset)
        random.shuffle(self.art_dataset)
        # self.dataset = images
        print('Finished preprocessing the Art Works dataset, total image number: %d...'%len(self.art_dataset))
        print('Finished preprocessing the Content dataset, total image number: %d...'%len(self.content_dataset))

    def __getitem__(self, index):
        """Return one image and its corresponding attribute label."""
        filename        = self.content_dataset[index]
        image           = Image.open(filename)
        content         = self.content_transform(image)
        art_index       = random.randint(0,self.art_num-1)
        filename,label  = self.art_dataset[art_index]
        image           = Image.open(filename)
        style           = self.style_transform(image)
        return content,style,label

    def __len__(self):
        """Return the number of images."""
        return self.num_images

def denorm(x):
    out = (x + 1) / 2
    return out.clamp_(0, 1)

def GetLoader(s_image_dir,c_image_dir, 
                style_selected_dir, content_selected_dir,
                crop_size=178, batch_size=16, num_workers=8, 
                colorJitterEnable=True, colorConfig={"brightness":0.05,"contrast":0.05,"saturation":0.05,"hue":0.05}):
    """Build and return a data loader."""
    
    s_transforms = []
    c_transforms = []
    
    s_transforms.append(T.Resize(768))
    # s_transforms.append(T.Resize(900))
    c_transforms.append(T.Resize(768))

    s_transforms.append(T.RandomCrop(crop_size,pad_if_needed=True,padding_mode='reflect'))
    c_transforms.append(T.RandomCrop(crop_size))

    s_transforms.append(T.RandomHorizontalFlip())
    c_transforms.append(T.RandomHorizontalFlip())
    
    s_transforms.append(T.RandomVerticalFlip())
    c_transforms.append(T.RandomVerticalFlip())

    if colorJitterEnable:
        if colorConfig is not None:
            print("Enable color jitter!")
            colorBrightness = colorConfig["brightness"]
            colorContrast   = colorConfig["contrast"]
            colorSaturation = colorConfig["saturation"]
            colorHue        = (-colorConfig["hue"],colorConfig["hue"])
            s_transforms.append(T.ColorJitter(brightness=colorBrightness,\
                                contrast=colorContrast,saturation=colorSaturation, hue=colorHue))
            c_transforms.append(T.ColorJitter(brightness=colorBrightness,\
                                contrast=colorContrast,saturation=colorSaturation, hue=colorHue))
    s_transforms.append(T.ToTensor())
    c_transforms.append(T.ToTensor())

    s_transforms.append(T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)))
    c_transforms.append(T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)))
    
    s_transforms = T.Compose(s_transforms)
    c_transforms = T.Compose(c_transforms)

    content_dataset = TotalDataset(c_image_dir,s_image_dir, content_selected_dir, style_selected_dir
                        , c_transforms,s_transforms)
    content_data_loader = data.DataLoader(dataset=content_dataset,batch_size=batch_size,
                    drop_last=True,shuffle=True,num_workers=num_workers,pin_memory=True)
    prefetcher = DataPrefetcher(content_data_loader)
    return prefetcher

def GetValiDataTensors(
                image_dir=None,
                selected_imgs=[],
                crop_size=178,
                mean = (0.5, 0.5, 0.5),
                std=(0.5, 0.5, 0.5)
            ):
            
    transforms = []
    
    transforms.append(T.Resize(768))

    transforms.append(T.RandomCrop(crop_size,pad_if_needed=True,padding_mode='reflect'))

    transforms.append(T.ToTensor())

    transforms.append(T.Normalize(mean=mean, std=std))
    
    transforms = T.Compose(transforms)

    result_img   = []
    print("Start to read validation data......")
    if len(selected_imgs) != 0:
        for s_img in selected_imgs:
            if image_dir == None:
                temp_img = s_img
            else:
                temp_img = os.path.join(image_dir, s_img)
            temp_img = Image.open(temp_img)
            temp_img = transforms(temp_img).cuda().unsqueeze(0)
            result_img.append(temp_img)
    else:
        s_imgs = glob.glob(os.path.join(image_dir, '*.jpg'))
        s_imgs = s_imgs + glob.glob(os.path.join(image_dir, '*.png'))
        for s_img in s_imgs:
            temp_img = os.path.join(image_dir, s_img)
            temp_img = Image.open(temp_img)
            temp_img = transforms(temp_img).cuda().unsqueeze(0)
            result_img.append(temp_img)
    print("Finish to read validation data......")
    print("Total validation images: %d"%len(result_img))
    return result_img

def ScanAbnormalImg(image_dir, selected_imgs):
    """Scan the dataset, this function is designed to exclude or remove the non-RGB images."""
    print("processing images...")
    subffix = "jpg"
    for dir_item in selected_imgs:
        join_path = Path(image_dir,dir_item)#.replace('/','_'))
        if join_path.exists():
            print("processing %s"%dir_item)
            images = join_path.glob('*.%s'%(subffix))
            for item in images:
                # print(str(item.name)[0:6])
                # temp = cv2.imread(str(item))
                temp = Image.open(item)
                # exclude the abnormal images
                if temp.mode!="RGB":
                    print(temp.mode)
                    print("Found one abnormal image!")
                    print(item)
                    os.remove(str(item))

        else:
            print("%s dir does not exist!"%dir_item)