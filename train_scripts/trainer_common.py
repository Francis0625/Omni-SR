#!/usr/bin/env python3
# -*- coding:utf-8 -*-
#############################################################
# File: trainer_common.py
# Created Date: Friday December 25th 2020
# Author: Chen Xuanhong
# Email: chenxuanhongzju@outlook.com
# Last Modified:  Wednesday, 19th April 2023 10:51:24 pm
# Modified By: Chen Xuanhong
# Copyright (c) 2020 Shanghai Jiao Tong University
#############################################################


import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F


from tqdm import tqdm
from utilities.utilities import calculate_psnr, calculate_ssim,tensor2img


# modify this template to derive your train class

class Trainer(object):
    def __init__(self, config, reporter):

        self.config     = config
        # logger
        self.reporter   = reporter
        # Data loader
        #============build train dataloader==============#
        # TODO to modify the key: "your_train_dataset" to get your train dataset path
        train_dataset   = config["dataset_paths"][config["dataset_name"]]
        #================================================#
        print("Prepare the train dataloader...")
        dlModulename    = config["dataloader"]
        package         = __import__("data_tools.dataloader_%s"%dlModulename, fromlist=True)
        dataloaderClass = getattr(package, 'GetLoader')
        
        dataloader      = dataloaderClass(train_dataset,
                                        config["batch_size"],
                                        config["random_seed"],
                                        **config["dataset_params"]
                                    )
        
        self.train_loader= dataloader
 
        #========build evaluation dataloader=============#
        # TODO to modify the key: "your_eval_dataset" to get your evaluation dataset path
        # eval_dataset = config["test_dataset_paths"][config["eval_dataset_name"].lower()]

        #================================================#
        print("Prepare the evaluation dataloader...")
        dlModulename    = config["eval_dataloader"]
        package         = __import__("data_tools.eval_dataloader_%s"%dlModulename, fromlist=True)
        dataloaderClass = getattr(package, 'EvalDataset')

        #================urban100======================#
        self.eval_loader1      = dataloaderClass('urban100',
                                        config["test_dataset_paths"]['urban100'],
                                        config["eval_batch_size"],
                                        image_scale = config["dataset_params"]["image_scale"]
                                        )
        self.eval_iter1  = len(self.eval_loader1)//config["eval_batch_size"]
        if len(self.eval_loader1)%config["eval_batch_size"]>0:
            self.eval_iter1+=1
        
        #================b100======================#
        self.eval_loader2      = dataloaderClass('b100',
                                        config["test_dataset_paths"]['b100'],
                                        config["eval_batch_size"],
                                        image_scale = config["dataset_params"]["image_scale"]
                                        )
        self.eval_iter2  = len(self.eval_loader2)//config["eval_batch_size"]
        if len(self.eval_loader2)%config["eval_batch_size"]>0:
            self.eval_iter2+=1

        #================set14======================#
        self.eval_loader3      = dataloaderClass('set14',
                                        config["test_dataset_paths"]['set14'],
                                        config["eval_batch_size"],
                                        image_scale = config["dataset_params"]["image_scale"]
                                        )
        self.eval_iter3  = len(self.eval_loader3)//config["eval_batch_size"]
        if len(self.eval_loader3)%config["eval_batch_size"]>0:
            self.eval_iter3+=1


        #================set5======================#
        self.eval_loader4      = dataloaderClass('set5',
                                        config["test_dataset_paths"]['set5'],
                                        config["eval_batch_size"],
                                        image_scale = config["dataset_params"]["image_scale"]
                                        )
        self.eval_iter4  = len(self.eval_loader4)//config["eval_batch_size"]
        if len(self.eval_loader4)%config["eval_batch_size"]>0:
            self.eval_iter4+=1


        #==============build tensorboard=================#
        if self.config["use_tensorboard"]:
            from utilities.utilities import build_tensorboard
            self.tensorboard_writer = build_tensorboard(self.config["project_summary"])


    # TODO modify this function to build your models
    def __init_framework__(self):
        '''
            This function is designed to define the framework,
            and print the framework information into the log file
        '''
        #===============build models================#
        print("build models...")
        # TODO [import models here]
        # from components.RepSR_plain import RepSRPlain_pixel
        script_name     = "components."+self.config["module_script_name"]
        class_name      = self.config["class_name"]
        package         = __import__(script_name, fromlist=True)
        network_class   = getattr(package, class_name)

        # print and recorde model structure
        self.reporter.writeInfo("Model structure:")

        # TODO replace below lines to define the model framework
        self.network = network_class(3,
                                    3,
                                    self.config["feature_num"],
                                    **self.config["module_params"]
                                    )
        self.reporter.writeModel(self.network.__str__())
        
        # train in GPU
        if self.config["cuda"] >=0:
            self.network = self.network.cuda()
        
        print(self.network)

        # if in finetune phase, load the pretrained checkpoint
        # if self.config["phase"] == "finetune":
        #     model_path = os.path.join(self.config["project_checkpoints"],
        #                                 "epoch%d_%s.pth"%(self.config["checkpoint_step"],
        #                                 self.config["checkpoint_names"]["generator_name"]))
        #     self.network.load_state_dict(torch.load(model_path))
        #     print('loaded trained backbone model epoch {}...!'.format(self.config["project_checkpoints"]))

        # if in imagenet finetune phase, load the pretrained checkpoint
        if self.config["phase"] == "finetune":
            model_path = os.path.join(self.config["imagenet_checkpoints"],
                                        "epoch%d_%s.pth"%(self.config["ckpt"],
                                        self.config["checkpoint_names"]["generator_name"]))
            self.network.load_state_dict(torch.load(model_path), strict=False)
            print('loaded trained backbone model epoch {}...!'.format(self.config["imagenet_checkpoints"]))


    # TODO modify this function to evaluate your model
    def __evaluation__(self, eval_loader, eval_iter, epoch, step = 0):
        # Evaluate the checkpoint
        self.network.eval()
        total_psnr = 0
        total_ssim = 0
        total_num  = 0
        dataset_name = eval_loader.dataset_name
        with torch.no_grad():
            for _ in tqdm(range(eval_iter)):
                hr, lr = eval_loader()
                if self.config["cuda"] >=0:
                    hr = hr.cuda()
                    lr = lr.cuda()
                res     = self.network(lr)
                res     = tensor2img(res.cpu())
                hr      = tensor2img(hr.cpu())
                psnr    = calculate_psnr(res[0],hr[0])
                ssim    = calculate_ssim(res[0],hr[0])
                total_psnr+= psnr
                total_ssim+= ssim
                total_num+=1
        final_psnr = total_psnr/total_num
        final_ssim = total_ssim/total_num
        # print("[{}], Epoch [{}], psnr: {:.4f}, ssim: {:.5f}".format(self.config["version"],
        #                                             epoch, final_psnr, final_ssim))
        # self.reporter.writeTrainLog(epoch,step,"psnr: {:.4f}, ssim: {:.5f}".format(final_psnr, final_ssim))
        # self.tensorboard_writer.add_scalar('metric/PSNR', final_psnr, epoch)
        # self.tensorboard_writer.add_scalar('metric/SSIM', final_ssim, epoch)
        if (dataset_name == "Urban100") and (final_psnr>self.best_psnr["psnr"]):
            self.best_psnr["psnr"] = final_psnr
            self.best_psnr["epoch"] = epoch

        print("[{}], Epoch [{}], Dataset: {}, PSNR: {:.4f}, SSIM: {:.4f}".format(self.config["version"], epoch, dataset_name, final_psnr, final_ssim))
        self.reporter.writeTrainLog(epoch,step, "Dataset: {}, PSNR: {:.4f}, SSIM: {:.4f}".format(dataset_name, final_psnr, final_ssim))

        if dataset_name == "Set5":
            self.tensorboard_writer.add_scalar('metric/PSNR_Set5', final_psnr, epoch)
        elif dataset_name == "Set14":
            self.tensorboard_writer.add_scalar('metric/PSNR_Set14', final_psnr, epoch)
        elif dataset_name == "B100":
            self.tensorboard_writer.add_scalar('metric/PSNR_B100', final_psnr, epoch)
        elif dataset_name == "Urban100":
            print("[{}], Best Urban100 PSNR: {:.4f} @ epoch {}".format(self.config["version"],
                                                    self.best_psnr["psnr"], self.best_psnr["epoch"]))
            self.tensorboard_writer.add_scalar('metric/PSNR_Urban100', final_psnr, epoch)
        else:
            raise FileNotFoundError

    # TODO modify this function to configurate the optimizer of your pipeline
    def __setup_optimizers__(self):
        train_opt = self.config['optim_config'] 
        optim_params = []
        for k, v in self.network.named_parameters():
            if v.requires_grad:
                optim_params.append(v)
            else:
                self.reporter.writeInfo(f'Params {k} will not be optimized.')

        optim_type = self.config['optim_type']
        if optim_type.lower() == 'adam':
            self.optimizer = torch.optim.Adam(optim_params,**train_opt)
        elif optim_type.lower() == 'adamw':
            self.optimizer = torch.optim.AdamW(optim_params,**train_opt)
        else:
            raise NotImplementedError(
                f'optimizer {optim_type} is not supperted yet.')
        # self.optimizers.append(self.optimizer_g)
        

    def train(self):
        
        # general configurations 
        ckpt_dir    = self.config["project_checkpoints"]
        log_frep    = self.config["log_step"]
        model_freq  = self.config["model_save_epoch"]
        total_epoch = self.config["total_epoch"]
        l1_W        = self.config["l1_weight"]
        # lrDecayStep = self.config["lrDecayStep"]
        # TODO [more configurations here]
        self.best_psnr = {
            "epoch":-1,
            "psnr":-1
        }

        #===============build framework================#
        self.__init_framework__()
        # import pdb; pdb.set_trace()

        from thop import profile
        from thop import clever_format
        train_patch_size = self.config["dataset_params"]["lr_patch_size"]
        test_img    = torch.rand((1,3,train_patch_size,train_patch_size)).cuda()

        macs, params = profile(self.network, inputs=(test_img,))
        macs, params = clever_format([macs, params], "%.3f")
        print("Model FLOPs: ",macs)
        print("Model Params:",params)
        self.reporter.writeInfo("Model FLOPs: "+macs)
        self.reporter.writeInfo("Model Params: "+params)

        # # set the start point for training loop
        # if self.config["phase"] == "finetune":
        #     start = self.config["checkpoint_epoch"] - 1
        # else:
        #     start = 0
        start = 0
        

        #===============build optimizer================#
        print("build the optimizer...")
        # Optimizer
        # TODO replace below lines to build your optimizer
        self.__setup_optimizers__()

        #===============build losses===================#
        # TODO replace below lines to build your losses
        l1 = nn.L1Loss() # [replace this]

        # from losses.PerceptualLoss import PerceptualLoss
        # perceptual_config = self.config["perceptual"]
        # ploss = PerceptualLoss(
        #                 perceptual_config["layer_weights"],
        #                 perceptual_config["vgg_type"],
        #                 perceptual_config["use_input_norm"],
        #                 perceptual_config["perceptual_weight"],
        #                 perceptual_config["criterion"]
        #             )
        # if self.config["cuda"] >=0:
        #     ploss = ploss.cuda()
        
        # Caculate the epoch number
        step_epoch  = len(self.train_loader)
        print("Total step = %d in each epoch"%step_epoch)

        # Start time
        import datetime
        print("Start to train at %s"%(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
        
        print('Start   ===========================  training...')
        start_time = time.time()
        # import pdb; pdb.set_trace()
        for epoch in range(start, total_epoch):
            for step in range(step_epoch):
                # Set the networks to train mode

                self.network.train()
                # TODO [add more code here]
                # clear cumulative gradient
                self.optimizer.zero_grad()

                # TODO read the training data
                
                hr, lr  = self.train_loader.next()
                
                
                generated_hr = self.network(lr)
                loss_l1  = l1(generated_hr, hr)

                loss_curr = loss_l1

                loss_curr.backward()

                self.optimizer.step()

                # caculate gradients

                # update weights
                # import pdb; pdb.set_trace()
                # self.optimizer.step()
                
                # Print out log info
                if (step + 1) % log_frep == 0:
                    elapsed = time.time() - start_time
                    elapsed = str(datetime.timedelta(seconds=elapsed))

                    # cumulative steps
                    cum_step = (step_epoch * epoch + step + 1)
                    
                    #==================Print log info======================#
                    # print("[{}], Elapsed [{}], Epoch [{}/{}], Step [{}/{}], loss: {:.4f}, l1: {:.4f}, perceptual: {:.4f}".
                    #     format(self.config["version"], elapsed, epoch + 1, total_epoch, step + 1, step_epoch, 
                    #             loss_curr.item(),loss_l1.item(), loss_per.item()))
                    print("[{}], Elapsed [{}], Epoch [{}/{}], Step [{}/{}], loss: {:.4f}, l1: {:.4f}".
                        format(self.config["version"], elapsed, epoch + 1, total_epoch, step + 1, step_epoch, 
                                loss_curr.item(),loss_l1.item()))
                    #===================Write log info into log file=======#
                    # self.reporter.writeTrainLog(epoch+1,step+1,
                    #             "loss: {:.4f}, l1: {:.4f}, perceptual: {:.4f}".format(loss_curr.item(),
                    #                                                     loss_l1.item(), loss_per.item()))
                    
                    self.reporter.writeTrainLog(epoch+1,step+1,
                                "loss: {:.4f}, l1: {:.4f}".format(loss_curr.item(), loss_l1.item()))

                    #==================Tensorboard=========================#
                    # write training information into tensorboard log files
                    if self.config["use_tensorboard"]:

                        # TODO replace  below lines to record the losses or metrics 
                        self.tensorboard_writer.add_scalar('data/loss', loss_curr.item(), cum_step)
                        self.tensorboard_writer.add_scalar('data/l1', loss_l1.item(), cum_step)

            
            #===============adjust learning rate============#
            if (epoch + 1) in self.config["lr_decay_step"] and self.config["lr_decay_enable"]:
                print("Learning rate decay")
                for p in self.optimizer.param_groups:
                    p['lr'] *= self.config["lr_decay"]
                    print("Current learning rate is %f"%p['lr'])

            #===============save checkpoints================#
            if (epoch+1) % model_freq==0:
                print("Save epoch %d model checkpoint!"%(epoch+1))
                torch.save(self.network.state_dict(),
                        os.path.join(ckpt_dir, 'epoch{}_{}.pth'.format(epoch + 1, 
                                    self.config["checkpoint_names"]["generator_name"])))

                self.__evaluation__(self.eval_loader1, self.eval_iter1, epoch+1)
                self.__evaluation__(self.eval_loader2, self.eval_iter2, epoch+1)
                self.__evaluation__(self.eval_loader3, self.eval_iter3, epoch+1)
                self.__evaluation__(self.eval_loader4, self.eval_iter4, epoch+1)