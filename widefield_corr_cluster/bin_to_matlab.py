# -*- coding: utf-8 -*-
"""
Created on Tue Apr 21 14:22:52 2015

This script loads LabView-decimated binary movies from the 
Orca Flash4.0 sCMOS camera (2048x2048 reduced to 512x512 and downsampled from 
100 to 50 Hz) and saves them as a matlab matrix

@author: mattv
"""

import numpy as np
#import scipy.io as io
import os

def importMov(path, frameNum): 
    dtype = np.dtype('<u2') # < is little endian, 2 for 16bit
    columnNum = 512  
    rowNum = 512                           
    exposureTime = 20.
    bytenum = frameNum*rowNum*columnNum
    if frameNum == -1: #load all
        bytenum = -1 
    openit = open(path, 'r+b')
    imageFile = np.fromfile(openit,dtype=dtype,count=bytenum)  
    openit.close()
    frameNum = imageFile.shape[0]/(rowNum*columnNum)
    # in case number of loaded frames does not match frameNum, recalculate frameNum.  This is mostly for debugging purposes
    print 'width =', str(columnNum), 'pixels'
    print 'height =', str(rowNum), 'pixels'
    print 'length =', str(frameNum), 'frame(s)'
    print 'exposure time =', str(exposureTime), 'ms'    
    imageFile = imageFile.reshape((frameNum,rowNum,columnNum))
    return imageFile, exposureTime


frameNum = 1000
basepath = r'\\aibsdata2\nc-ophys\Matt\corrdata\150225_M166910'
filename = '20150225JCamF102deci244'
filepath = os.path.join(basepath, filename)

savename = r'C:\Users\mattv\Desktop\test.mat'

mov, exposuretime = importMov(filepath, frameNum)

#mov = mov.astype('float') #make sure you have enough RAM for this conversion

#io.savemat(savename, mdict={'mov': mov})