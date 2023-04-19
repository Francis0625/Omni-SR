#!/usr/bin/env python3
# -*- coding:utf-8 -*-
#############################################################
# File: tester_Matlab.py
# Created Date: Saturday February 6th 2021
# Author: Chen Xuanhong
# Email: chenxuanhongzju@outlook.com
# Last Modified:  Thursday, 20th April 2023 1:45:45 am
# Modified By: Chen Xuanhong
# Copyright (c) 2021 Shanghai Jiao Tong University
#############################################################


import  os
import  time

import  torch
import  torch.nn as nn
import  cv2
from    utilities.utilities import calculate_psnr, calculate_ssim, tensor2img

# from utilities.Reporter import Reporter
from    tqdm import tqdm

class Tester(object):
    def __init__(self, config, reporter):

        self.config     = config
        # logger
        self.reporter   = reporter

        #============build evaluation dataloader==============#
        print("Prepare the test dataloader...")
        # dlModulename    = config["test_dataloader"]
        dlModulename    = "rcan"
        package         = __import__("data_tools.test_dataloader_%s"%dlModulename, fromlist=True)
        dataloaderClass = getattr(package, 'TestDataset')
        dataloader      = dataloaderClass(
                                        config["test_dataset_names"],
                                        config["test_dataset_path"],
                                        config["batch_size"],
                                        config["dataset_params"]["degradation"],
                                        config["dataset_params"]["image_scale"],
                                        config["dataset_params"]["subffix"])
        self.test_loader= dataloader

        self.dataset_name= config["test_dataset_name"]
        self.image_scale = config["dataset_params"]["image_scale"]

        self.test_iter  = len(dataloader)


    def __init_framework__(self):
        '''
            This function is designed to define the framework,
            and print the framework information into the log file
        '''
        #===============build models================#
        print("build models...")
        # TODO [import models here]
        script_name     = "components."+self.config["module_script_name"]
        class_name      = self.config["class_name"]
        package         = __import__(script_name, fromlist=True)
        network_class   = getattr(package, class_name)
        # from components.FSRCNN import FSRCNN

        # TODO replace below lines to define the model framework        
        self.network = network_class(3,
                                    3,
                                    self.config["feature_num"],
                                    **self.config["module_params"]
                            )

        # print and recorde model structure
        self.reporter.writeInfo("Model structure:")
        self.reporter.writeModel(self.network.__str__())

        # train in GPU
        if self.config["cuda"] >=0:
            self.network = self.network.cuda()
        model_path = os.path.join(self.config["project_checkpoints"],
                                        "epoch%d_%s.pth"%(self.config["checkpoint_epoch"],
                                        self.config["checkpoint_names"]["generator_name"]))
        model_spec = torch.load(model_path, map_location=torch.device("cpu"))
        own_state = self.network.state_dict()
        # print(own_state.keys())
        for name, param in model_spec.items():
            if name in own_state:
                if isinstance(param, nn.Parameter):
                    param = param.data
                try:
                    own_state[name].copy_(param)
                except Exception:
                    if name.find('tail') == -1:
                        raise RuntimeError('While copying the parameter named {}, '
                                           'whose dimensions in the model are {} and '
                                           'whose dimensions in the checkpoint are {}.'
                                           .format(name, own_state[name].size(), param.size()))

        # print("Encoder pretrained weight loaded.....................")

        # self.network.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
        # self.network.load_state_dict(torch.load(pathwocao))
        print('loaded trained backbone model epoch {}...!'.format(self.config["checkpoint_epoch"]))

        if self.config["cuda"] >=0:
            self.network = self.network.cuda()
        # self.network.set_dropone()

    def test(self):

        # save_result = self.config["saveTestResult"]
        save_dir    = "./SR/"
        save_dir    = os.path.join(save_dir, "BI", self.config["version"],
                            self.config["test_dataset_names"], "x%d"%self.config["dataset_params"]["image_scale"])
        # save_dir      ="./demo"
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        # ckp_epoch   = self.config["checkpoint_epoch"]
        version     = self.config["version"]

        # models
        self.__init_framework__()

        # Start time
        import datetime
        print("Start to test at %s"%(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
        print('Start ===================================  test...')
        start_time = time.time()
        total_psnr = 0
        total_ssim = 0
        total_num  = 0
        self.network.eval()
        patch_test = True
        with torch.no_grad():
            for iii in tqdm(range(self.test_iter)):
                hr,lr,names = self.test_loader()
                # print(lr.shape)
                if self.config["cuda"] >=0:
                    hr = hr.cuda()
                    lr = lr.cuda()
                if patch_test:
                    tile = 64
                    tile_overlap = 24
                    scale = self.config["module_params"]["upsampling"]
                    b, c, h, w = lr.size()
                    tile = min(tile, h, w)

                    stride = tile - tile_overlap
                    h_idx_list = list(range(0, h-tile, stride)) + [h-tile]
                    w_idx_list = list(range(0, w-tile, stride)) + [w-tile]
                    E = torch.zeros(b, c, h*scale, w*scale).type_as(lr)
                    W = torch.zeros_like(E)

                    for h_idx in h_idx_list:
                        for w_idx in w_idx_list:
                            in_patch = lr[..., h_idx:h_idx+tile, w_idx:w_idx+tile]
                            out_patch = self.network(in_patch)
                            if isinstance(out_patch, list):
                                out_patch = out_patch[-1]
                            out_patch_mask = torch.ones_like(out_patch)

                            E[..., h_idx*scale:(h_idx+tile)*scale, w_idx*scale:(w_idx+tile)*scale].add_(out_patch)
                            W[..., h_idx*scale:(h_idx+tile)*scale, w_idx*scale:(w_idx+tile)*scale].add_(out_patch_mask)
                    res = E.div_(W)
                else:
                    res = self.network(lr)
 
                dataset_size = res.shape[0]

                res = tensor2img(res.cpu())
                hr  = tensor2img(hr.cpu())

               

                for t in range(dataset_size):
                    temp_img = res[t,:,:,:]
                    psnr     = calculate_psnr(temp_img, hr[t,:,:,:])
                    ssim     = calculate_ssim(temp_img, hr[t,:,:,:])
                    # print("PSNR is %.3f"%psnr)
                    temp_img = cv2.cvtColor(temp_img,cv2.COLOR_RGB2BGR)
                    i_name = names[t]
                    cv2.imwrite(os.path.join(save_dir,'{}.png'.format(i_name).replace('LRBI',version)),temp_img)
                    total_num += 1
                    total_psnr += psnr
                    total_ssim += ssim
            final_psnr = total_psnr/total_num
            final_ssim = total_ssim/total_num          

        elapsed = time.time() - start_time
        elapsed = str(datetime.timedelta(seconds=elapsed))
        print("Elapsed [{}], PSNR: {:.4f}, SSIM: {:.4f}".format(elapsed,final_psnr, final_ssim))
        self.reporter.writeInfo("Plain Checkpoint Epoch: {}, PSNR: {:.4f}, SSIM: {:.4f}".format(
                                        self.config["checkpoint_epoch"], final_psnr, final_ssim))