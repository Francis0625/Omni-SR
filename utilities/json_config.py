#!/usr/bin/env python3
# -*- coding:utf-8 -*-
#############################################################
# File: json_config.py
# Created Date: Tuesday April 28th 2022
# Author: Chen Xuanhong
# Email: chenxuanhongzju@outlook.com
# Last Modified:  Thursday, 20th April 2023 9:29:51 am
# Modified By: Chen Xuanhong
# Copyright (c) 2020 Shanghai Jiao Tong University
#############################################################

import  json


def readConfig(path):
    with open(path,'r') as cf:
        nodelocaltionstr = cf.read()
        nodelocaltioninf = json.loads(nodelocaltionstr)
        if isinstance(nodelocaltioninf,str):
            nodelocaltioninf = json.loads(nodelocaltioninf)
    return nodelocaltioninf

def writeConfig(path, info):
    with open(path, 'w') as cf:
        configjson  = json.dumps(info, indent=4)
        cf.writelines(configjson)