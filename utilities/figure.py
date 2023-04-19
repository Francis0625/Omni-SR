#!/usr/bin/env python3
# -*- coding:utf-8 -*-
#############################################################
# File: figure.py
# Created Date: Tuesday October 13th 2020
# Author: Chen Xuanhong
# Email: chenxuanhongzju@outlook.com
# Last Modified:  Tuesday, 13th October 2020 2:54:30 pm
# Modified By: Chen Xuanhong
# Copyright (c) 2020 Shanghai Jiao Tong University
#############################################################

import matplotlib.pyplot as plt

def plot_loss_curve(losses, save_path):
    for key in losses.keys():
        plt.plot(range(len(losses[key])), losses[key], label=key)
    plt.xlabel('iteration')
    plt.title(f'loss curve')
    plt.legend()
    plt.savefig(save_path)
    plt.clf()