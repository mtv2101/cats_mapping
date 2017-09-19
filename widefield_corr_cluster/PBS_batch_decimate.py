#!/shared/utils.x86_64/python-2.7/bin/python
"""
Created on Thu Oct 29 10:08:50 2015

@author: mattv
"""

import os
import time
import sys
import stat

if __name__ == "__main__":
            
    # M187476
    paths = [
    r'/data/nc-ophys/CorticalMapping/IntrinsicImageData/150720-M187476/20150720JCamF109_2_2_1.npy',
    # r'/data/nc-ophys/CorticalMapping/IntrinsicImageData/150722-M187476/150722JCamF103_2_2_1.npy',
    # r'/data/nc-ophys/CorticalMapping/IntrinsicImageData/150723-M187476/010101JCamF102_2_2_1.npy',
    #r'/data/nc-ophys/CorticalMapping/IntrinsicImageData/150727-M187476/150727JCamF102_2_2_1.npy',
    #r'/data/nc-ophys/CorticalMapping/IntrinsicImageData/150729-M187476/010101JCamF102_2_2_1.npy',
    # r'/data/nc-ophys/CorticalMapping/IntrinsicImageData/150731-M187476/010101JCamF102_2_2_1.npy',
    
    # M187474
    # r'/data/nc-ophys/CorticalMapping/IntrinsicImageData/150720-M187474/20150720JCamF104'
    # r'/data/nc-ophys/CorticalMapping/IntrinsicImageData/150723-M187474/010101JCamF102_2_2_1.npy',
    # r'/data/nc-ophys/CorticalMapping/IntrinsicImageData/150728-M187474/010101JCamF102_2_2_1.npy',
    #r'/data/nc-ophys/CorticalMapping/IntrinsicImageData/150730-M187474/010101JCamF102_2_2_1.npy',
    #r'/data/nc-ophys/CorticalMapping/IntrinsicImageData/150804-M187474/010101JCamF102_2_2_1.npy',
    r'/data/nc-ophys/CorticalMapping/IntrinsicImageData/150806-M187474/010101JCamF104_2_2_1.npy',
    # r'/data/nc-ophys/CorticalMapping/IntrinsicImageData/150807-M187474/010101JCamF102_2_2_1.npy',
    # r'/data/nc-ophys/CorticalMapping/IntrinsicImageData/150810-M187474/010101JCamF102_2_2_1.npy',
    
    # M187201
    #r'/data/nc-ophys/CorticalMapping/IntrinsicImageData/150720-M187201/20150720JCamF100_2_2_1.npy',
    # r'/data/nc-ophys/CorticalMapping/IntrinsicImageData/150721-M187201/20150721JCamF101'
    # r'/data/nc-ophys/CorticalMapping/IntrinsicImageData/150723-M187201/010101JCamF103_2_2_1.npy',
    #r'/data/nc-ophys/CorticalMapping/IntrinsicImageData/150724-M187201/010101JCamF102_2_2_1.npy',
    #r'/data/nc-ophys/CorticalMapping/IntrinsicImageData/150727-M187201/010101JCamF102_2_2_1.npy',
    r'/data/nc-ophys/CorticalMapping/IntrinsicImageData/150729-M187201/010101JCamF102_2_2_1.npy',
    # r'/data/nc-ophys/CorticalMapping/IntrinsicImageData/150730-M187201/010101JCamF105_2_2_1.npy',
    # r'/data/nc-ophys/CorticalMapping/IntrinsicImageData/150803-M187201/150803JCamF102_2_2_1.npy',
    # r'/data/nc-ophys/CorticalMapping/IntrinsicImageData/150805-M187201/010101JCamF102_2_2_1.npy',
    # r'/data/nc-ophys/CorticalMapping/IntrinsicImageData/150807-M187201/010101JCamF102_2_2_1.npy',
    # r'/data/nc-ophys/CorticalMapping/IntrinsicImageData/150810-M187201/010101JCamF102_2_2_1.npy',
    
    #M177931
    # r'/data/nc-ophys/CorticalMapping/IntrinsicImageData/150701-M177931/20150701JCamF102deci1044.npy',
    # r'/data/nc-ophys/CorticalMapping/IntrinsicImageData/150702-M177931/20150702JCamF104_4_4_1.npy',
    # r'/data/nc-ophys/CorticalMapping/IntrinsicImageData/150703-M177931/20150703JCamF104_4_4_1.npy',
    #r'/data/nc-ophys/CorticalMapping/IntrinsicImageData/150704-M177931/20150704JCamF103_4_4_1.npy',
    #r'/data/nc-ophys/CorticalMapping/IntrinsicImageData/150705-M177931/20150705JCamF105_4_4_1.npy',
    r'/data/nc-ophys/CorticalMapping/IntrinsicImageData/150706-M177931/20150706JCamF103_2_2_1.npy',
    #r'/data/nc-ophys/CorticalMapping/IntrinsicImageData/150722-M177931/010101JCamF103_2_2_1.npy',
    #r'/data/nc-ophys/CorticalMapping/IntrinsicImageData/150724-M177931/010101JCamF102_2_2_1.npy',
    # r'/data/nc-ophys/CorticalMapping/IntrinsicImageData/150727-M177931/010101JCamF103_2_2_1.npy',
    #r'/data/nc-ophys/CorticalMapping/IntrinsicImageData/150729-M177931/010101JCamF102_2_2_1.npy',
    #r'/data/nc-ophys/CorticalMapping/IntrinsicImageData/150731-M177931/010101JCamF102_2_2_1.npy',
    r'/data/nc-ophys/CorticalMapping/IntrinsicImageData/150805-M177931/010101JCamF102_2_2_1.npy',
    r'/data/nc-ophys/CorticalMapping/IntrinsicImageData/150807-M177931/010101JCamF102_2_2_1.npy',

    
    #M177929
    #r'/data/nc-ophys/CorticalMapping/IntrinsicImageData/150701-M177929/20150701JCamF101deci1044'
    # r'/data/nc-ophys/CorticalMapping/IntrinsicImageData/150702-M177929/20150702JCamF101_4_4_1.npy',
    # r'/data/nc-ophys/CorticalMapping/IntrinsicImageData/150703-M177929/20150703JCamF102_4_4_1.npy',
    # r'/data/nc-ophys/CorticalMapping/IntrinsicImageData/150704-M177929/20150704JCamF101_4_4_1.npy',
    # r'/data/nc-ophys/CorticalMapping/IntrinsicImageData/150705-M177929/20150705JCamF102_4_4_1.npy',
    # r'/data/nc-ophys/CorticalMapping/IntrinsicImageData/150706-M177929/20150706JCamF101_4_4_1.npy',
    #r'/data/nc-ophys/CorticalMapping/IntrinsicImageData/150722-M177929/150722JCamF102_2_2_1.npy',
    #r'/data/nc-ophys/CorticalMapping/IntrinsicImageData/150724-M177929/010101JCamF103_2_2_1.npy',
    # r'/data/nc-ophys/CorticalMapping/IntrinsicImageData/150728-M177929/010101JCamF102_2_2_1.npy',
    # r'/data/nc-ophys/CorticalMapping/IntrinsicImageData/150729-M177929/010101JCamF102_2_2_1.npy',
    # r'/data/nc-ophys/CorticalMapping/IntrinsicImageData/150730-M177929/010101JCamF102_2_2_1.npy',
    # r'/data/nc-ophys/CorticalMapping/IntrinsicImageData/150803-M177929/010101JCamF102_2_2_1.npy',
    r'/data/nc-ophys/CorticalMapping/IntrinsicImageData/150806-M177929/010101JCamF102_2_2_1.npy'

    ]
    
    test_paths = [
        r'/data/nc-ophys/CorticalMapping/IntrinsicImageData/150806-M187474/010101JCamF104_16_16_1.npy',
        r'/data/nc-ophys/CorticalMapping/IntrinsicImageData/150806-M187476/010101JCamF103_16_16_1.npy'
        ]

    somemorepaths = ['/data/nc-ophys/CorticalMapping/IntrinsicImageData/150724-M187478/010101JCamF102_2_2_1.npy',
      '/data/nc-ophys/CorticalMapping/IntrinsicImageData/150728-M187478/010101JCamF102_2_2_1.npy',
      '/data/nc-ophys/CorticalMapping/IntrinsicImageData/150729-M187478/010101JCamF102_2_2_1.npy',
      '/data/nc-ophys/CorticalMapping/IntrinsicImageData/150730-M187478/010101JCamF102_2_2_1.npy',
      '/data/nc-ophys/CorticalMapping/IntrinsicImageData/150731-M187478/010101JCamF102_2_2_1.npy',
      '/data/nc-ophys/CorticalMapping/IntrinsicImageData/150804-M187478/010101JCamF102_2_2_1.npy',
      '/data/nc-ophys/CorticalMapping/IntrinsicImageData/150805-M187478/010101JCamF102_2_2_1.npy']

    
    id_tag = time.strftime("%Y%m%d-%H%M%S") # timestamp identifies the job

    def run_jobs(path, k, id_tag):        
        
        basepath, filename = os.path.split(path)
        base, mousepath = os.path.split(basepath)
        savepath = r'/data/nc-ophys/Matt/corrdata/batch_out'
        id_tag = id_tag + '_deci'
        save_path = os.path.join(savepath, id_tag)        
        
        if os.path.isdir(save_path) == False:
            os.mkdir(save_path)
        
        base_fname = str(mousepath) + '_' + str(id_tag)
        base_fpath = os.path.join(save_path, base_fname)
        
        shell_save_path = r'/data/nc-ophys/Matt/batch_scripts'
        shell_fname = base_fname + '.py'
        shell_fpath = os.path.join(shell_save_path, shell_fname)
        
        file = open(shell_fpath, 'w+') 
        file.writelines( ['import sys\n' ,
                          'sys.path.append(r\'/data/nc-ophys/Matt/\')\n' ,
                          'from aibs.CorticalMapping.widefield_corr_cluster.slicer_to_pytable import slicer_to_pytable as deci\n', 
                          'deci(\'' + path + '\', \'' + base_fpath + '\', 4)',      
                          #'print \'test\''               
                          ] )
        file.close()  
        time.sleep(.5)
        
        jobname = 'Jcorr_batch_deci%d' % k
        file = open(jobname + '.sub', 'w+')
        file.writelines( [ '#!/bin/sh\n',
                           '#PBS -N %s\n' % jobname ,
                           '#PBS -o %s\n' % (base_fpath + '.out') ,
                           '#PBS -q uno\n' ,
                           '#PBS -j oe\n' ,
                           '#PBS -l nodes=1:ppn=1\n' ,
                           '#PBS -l walltime=12:00:00\n' ,
                           '#PBS -l vmem=8g\n' ,
                           '#PBS -m a\n' ,
                           '#PBS -r n\n' ,
                           'cd $PBS_O_WORKDIR\n' ,
                           '\n' ,
                           '/shared/utils.x86_64/python-2.7/bin/python ' + shell_fpath + ' > %s\n' % (base_fpath + '.txt')
                           ] ) 
        file.close()
    
        os.system('qsub %s' % (jobname + '.sub')) 
    
        time.sleep(.5)

    [run_jobs(path, k, id_tag) for k,path in enumerate(somemorepaths)]    