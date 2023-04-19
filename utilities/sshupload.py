#!/usr/bin/env python3
# -*- coding:utf-8 -*-
#############################################################
# File: sshupload.py
# Created Date: Tuesday September 24th 2019
# Author: Lcx
# Email: chenxuanhongzju@outlook.com
# Last Modified:  Thursday, 30th June 2022 11:08:22 am
# Modified By: Chen Xuanhong
# Copyright (c) 2019 Shanghai Jiao Tong University
#############################################################

try:
    import paramiko
except:
    from pip._internal import main
    main(['install', 'paramiko'])
    import paramiko
import os
from pathlib import Path
# ssh传输类：

class fileUploaderClass(object):
    def __init__(self,serverIp,userName,passWd,port=22):
        self.__ip__         = serverIp
        self.__userName__   = userName
        self.__passWd__     = passWd
        self.__port__       = port
        self.__ssh__        = paramiko.SSHClient()
        self.__ssh__.set_missing_host_key_policy(paramiko.AutoAddPolicy())

    def sshScpPut(self,localFile,remoteFile):
        self.__ssh__.connect(self.__ip__, self.__port__ , self.__userName__, self.__passWd__)
        sftp = paramiko.SFTPClient.from_transport(self.__ssh__.get_transport())
        sftp = self.__ssh__.open_sftp()
        remoteDir  = remoteFile.split("/")
        if remoteFile[0]=='/':
            sftp.chdir('/')
            
        for item in remoteDir[0:-1]:
            if item == "":
                continue
            try:
                sftp.chdir(item)
            except:
                sftp.mkdir(item)
                sftp.chdir(item)
        sftp.put(localFile,remoteDir[-1])
        sftp.close()
        self.__ssh__.close()
        print("ssh localfile:%s remotefile:%s success"%(localFile,remoteFile))

    def sshScpGetNames(self,remoteDir):
        self.__ssh__.connect(self.__ip__, self.__port__ , self.__userName__, self.__passWd__)
        sftp = paramiko.SFTPClient.from_transport(self.__ssh__.get_transport())
        sftp = self.__ssh__.open_sftp()
        wocao = sftp.listdir(remoteDir)
        return wocao
    
    def sshScpGetDir(self, remoteDir, localDir, showProgress=False):
        self.__ssh__.connect(self.__ip__, self.__port__, self.__userName__, self.__passWd__)
        sftp = paramiko.SFTPClient.from_transport(self.__ssh__.get_transport())
        sftp = self.__ssh__.open_sftp()
        try:
            sftp.stat(remoteDir)
            print("Remote dir exists!")
        except:
            print("Remote dir does not exist!")
            return False
        files = sftp.listdir(remoteDir)
        for i_f in files:
            i_remote_file = Path(remoteDir,i_f).as_posix()
            local_file    = Path(localDir,i_f)
            if showProgress:
                sftp.get(i_remote_file, local_file,callback=self.__putCallBack__)
            else:
                sftp.get(i_remote_file, local_file)
        sftp.close()
        self.__ssh__.close()
        return True
    
    def sshScpGet(self, remoteFile, localFile, showProgress=False):
        self.__ssh__.connect(self.__ip__, self.__port__, self.__userName__, self.__passWd__)
        sftp = paramiko.SFTPClient.from_transport(self.__ssh__.get_transport())
        try:
            sftp.stat(remoteFile)
            print("Remote file exists!")
        except:
            print("Remote file does not exist!")
            return False
        sftp = self.__ssh__.open_sftp()
        if showProgress:
            sftp.get(remoteFile, localFile,callback=self.__putCallBack__)
        else:
            sftp.get(remoteFile, localFile)
        sftp.close()
        self.__ssh__.close()
        return True
    
    def __putCallBack__(self,transferred,total):
        print("current transferred %3.1f percent"%(transferred/total*100),end='\r')
    
    def sshScpGetmd5(self, file_path):
        self.__ssh__.connect(self.__ip__, self.__port__, self.__userName__, self.__passWd__)
        sftp = paramiko.SFTPClient.from_transport(self.__ssh__.get_transport())
        sftp = self.__ssh__.open_sftp() 
        try:
            file = sftp.open(file_path, 'rb')
            res  = (True,hashlib.new('md5', file.read()).hexdigest())
            sftp.close()
            self.__ssh__.close()
            return res
        except:
            sftp.close()
            self.__ssh__.close()
            return (False,None)
            
    def sshScpRename(self, oldpath, newpath):
        self.__ssh__.connect(self.__ip__, self.__port__ , self.__userName__, self.__passWd__)
        sftp = paramiko.SFTPClient.from_transport(self.__ssh__.get_transport())
        sftp = self.__ssh__.open_sftp()
        sftp.rename(oldpath,newpath)
        sftp.close()
        self.__ssh__.close()
        print("ssh oldpath:%s newpath:%s success"%(oldpath,newpath))

    def sshScpDelete(self,path):
        self.__ssh__.connect(self.__ip__, self.__port__ , self.__userName__, self.__passWd__)
        sftp = paramiko.SFTPClient.from_transport(self.__ssh__.get_transport())
        sftp = self.__ssh__.open_sftp()
        sftp.remove(path)
        sftp.close()
        self.__ssh__.close()
        print("ssh delete:%s success"%(path))
    
    def sshScpDeleteDir(self,path):
        self.__ssh__.connect(self.__ip__, self.__port__ , self.__userName__, self.__passWd__)
        sftp = paramiko.SFTPClient.from_transport(self.__ssh__.get_transport())
        sftp = self.__ssh__.open_sftp()
        self.__rm__(sftp,path)
        sftp.close()
        self.__ssh__.close()
        
    def __rm__(self,sftp,path):
        try:
            files = sftp.listdir(path=path)
            print(files)
            for f in files:
                filepath = os.path.join(path, f).replace('\\','/')
                self.__rm__(sftp,filepath)
            sftp.rmdir(path)
            print("ssh delete:%s success"%(path))
        except:
            print(path)
            sftp.remove(path)
            print("ssh delete:%s success"%(path))