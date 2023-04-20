#!/usr/bin/env python3
# -*- coding:utf-8 -*-
#############################################################
# File: yaml_config_.py
# Created Date: Monday February 17th 2020
# Author: Chen Xuanhong
# Email: chenxuanhongzju@outlook.com
# Last Modified:  Thursday, 20th April 2023 9:30:31 am
# Modified By: Chen Xuanhong
# Copyright (c) 2020 Shanghai Jiao Tong University
#############################################################


import yaml

def getConfigYaml(yaml_file):
    with open(yaml_file, 'r') as config_file:
        try:
            config_dict = yaml.load(config_file, Loader=yaml.FullLoader)
            return config_dict
        except ValueError:
            print('INVALID YAML file format.. Please provide a good yaml file')
            exit(-1)

if __name__ == "__main__":
    a= getConfigYaml("./train_256.yaml")
    sys_state = {}
    for item in a.items():
        sys_state[item[0]] = item[1]