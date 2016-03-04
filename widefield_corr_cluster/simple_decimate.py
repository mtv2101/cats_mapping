#!/shared/utils.x86_64/python-2.7/bin/python

"""
Created on Fri Jun 12 10:54:40 2015

@author: mattv
"""

import os
import platform
import datetime
import numpy as np

if platform.system() == 'Windows':
    basepath = r'\\aibsdata2\nc-ophys\Matt\corrdata\150225_M166910' 
    
elif platform.system() == 'Linux':  
    basepath = r'/data/nc-ophys/Matt/retinotopic_mapping/dome_20150611/'
    
filename = '20150225JCamF102deci244' #file to open
filepath = os.path.join(basepath, filename)

frameNum = 1000
dsfactor = 1
columnNum = 512  
rowNum = 512

timestamp_str = str(datetime.datetime.now())

def importMov(path, frameNum): 
    dtype = np.dtype('<u2') # < is little endian, 2 for 16bit                           
    exposureTime = 10.
    headerLength = 0#242
    bytenum = frameNum*rowNum*columnNum
    if frameNum == -1: #load all
        bytenum = -1 
    openit = open(path, 'r+b')
    imageFile = np.fromfile(openit,dtype=dtype,count=bytenum)  
    openit.close()
    frames = imageFile.shape[0]/(rowNum*columnNum)
    # in case number of loaded frames does not match frameNum, recalculate frameNum.  This is mostly for debugging purposes
    print 'width =', str(columnNum), 'pixels'
    print 'height =', str(rowNum), 'pixels'
    print 'length =', str(frameNum), 'frame(s)'
    print 'exposure time =', str(exposureTime), 'ms'  
    imageFile = imageFile[headerLength:] 
    imageFile = imageFile.reshape((frames,rowNum,columnNum))
    return imageFile, frames
    
def sub_sample_movie(mov, dsfactor):
    print 'subsampling movie...'
    mov_tiny = np.zeros([mov.shape[0],mov.shape[1]/dsfactor, mov.shape[2]/dsfactor], dtype=('u2'))
    for i in xrange(mov_tiny.shape[-2]):
        for j in xrange(mov_tiny.shape[-1]):
            mov_tiny[:,i,j] = mov[:, i*dsfactor:(i+1)*dsfactor,j*dsfactor:(j+1)*dsfactor].mean(-1).mean(-1)
    return mov_tiny
    
mov, frames = importMov(filepath, frameNum)

mov_tiny = sub_sample_movie(mov, dsfactor)
#mov_tiny = mov_tiny.astype('u2')

mov_tiny = np.reshape(mov_tiny, [frameNum, columnNum*rowNum])

movsavepath = os.path.join(basepath,filename+'_rawmovie_lin_'+timestamp_str[:10])
np.save(movsavepath, mov_tiny)