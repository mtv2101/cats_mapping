#!/shared/utils.x86_64/python-2.7/bin/python
"""
Created on Thu Oct 29 10:08:50 2015

@author: mattv
"""

import os
import time
import sys
import stat
import fnmatch      


if __name__ == "__main__":

	fnames = [
		 r'20150703JCamF102_4_4_1_128x128_150703-M177929_20160210-100558_deci.h5',
         r'20150703JCamF104_4_4_1_128x128_150703-M177931_20160210-100558_deci.h5',
         r'20150704JCamF101_4_4_1_128x128_150704-M177929_20160210-100558_deci.h5',
         r'20150705JCamF102_4_4_1_128x128_150705-M177929_20160210-100558_deci.h5',
         r'20150706JCamF101_4_4_1_128x128_150706-M177929_20160210-100558_deci.h5',
         r'20150702JCamF101_4_4_1_128x128_150702-M177929_20160210-100558_deci.h5',
         r'20150702JCamF104_4_4_1_128x128_150702-M177931_20160210-100558_deci.h5',
         r'20150704JCamF103_4_4_1_128x128_150704-M177931_20160210-110403_deci.h5',
         r'20150705JCamF105_4_4_1_128x128_150705-M177931_20160210-110403_deci.h5',
         r'010101JCamF102_2_2_1_128x128_150807-M177931_20160210-111828_deci.h5',
         r'010101JCamF102_2_2_1_128x128_150805-M177931_20160210-111828_deci.h5',
         r'20150706JCamF103_2_2_1_128x128_150706-M177931_20160210-111828_deci.h5',
         r'20150720JCamF109_2_2_1_128x128_150720-M187476_20160210-111828_deci.h5',
         r'010101JCamF102_2_2_1_128x128_150729-M187201_20160210-111828_deci.h5',
         r'010101JCamF102_2_2_1_128x128_150806-M177929_20160210-111828_deci.h5',
         r'010101JCamF104_2_2_1_128x128_150806-M187474_20160210-111828_deci.h5',
         r'010101JCamF102_2_2_1_128x128_150728-M177929_20151103-144358_deci.h5',
         r'010101JCamF102_2_2_1_128x128_150724-M177931_20151103-144358_deci.h5',
         r'010101JCamF102_2_2_1_128x128_150728-M187474_20151103-144358_deci.h5',
         r'010101JCamF102_2_2_1_128x128_150729-M177931_20151103-144358_deci.h5',
         r'010101JCamF102_2_2_1_128x128_150731-M177931_20151103-144358_deci.h5',
         r'010101JCamF102_2_2_1_128x128_150731-M187476_20151103-144358_deci.h5',
         r'010101JCamF102_2_2_1_128x128_150804-M187474_20151103-144358_deci.h5',
         r'010101JCamF102_2_2_1_128x128_150805-M187201_20151103-144358_deci.h5',
         r'010101JCamF102_2_2_1_128x128_150807-M187201_20151103-144358_deci.h5',
         r'010101JCamF102_2_2_1_128x128_150807-M187474_20151103-144358_deci.h5',
         r'010101JCamF102_2_2_1_128x128_150810-M187201_20151103-144358_deci.h5',
         r'010101JCamF102_2_2_1_128x128_150810-M187474_20151103-144358_deci.h5',
         r'010101JCamF105_2_2_1_128x128_150730-M187201_20151103-144358_deci.h5',
         r'150722JCamF103_2_2_1_128x128_150722-M187476_20151103-144358_deci.h5',
         r'150727JCamF102_2_2_1_128x128_150727-M187476_20151103-144358_deci.h5',
         r'010101JCamF102_2_2_1_128x128_150730-M177929_20151103-144358_deci.h5',
         r'010101JCamF102_2_2_1_128x128_150729-M187476_20151103-144358_deci.h5',
         r'150803JCamF102_2_2_1_128x128_150803-M187201_20151103-144358_deci.h5',
         r'010101JCamF102_2_2_1_128x128_150723-M187474_20151103-144358_deci.h5',
         r'010101JCamF102_2_2_1_128x128_150727-M187201_20151103-144358_deci.h5',
         r'010101JCamF103_2_2_1_128x128_150727-M177931_20151103-144358_deci.h5',
         r'010101JCamF102_2_2_1_128x128_150803-M177929_20151103-144358_deci.h5',
         r'010101JCamF102_2_2_1_128x128_150724-M187201_20151103-144358_deci.h5',
         r'010101JCamF103_2_2_1_128x128_150722-M177931_20151103-144358_deci.h5',
         r'010101JCamF103_2_2_1_128x128_150724-M177929_20151103-144358_deci.h5',
         r'150722JCamF102_2_2_1_128x128_150722-M177929_20151103-144358_deci.h5',
         r'010101JCamF102_2_2_1_128x128_150723-M187476_20151103-144358_deci.h5',
         r'20150720JCamF100_2_2_1_128x128_150720-M187201_20151103-144358_deci.h5',
         r'010101JCamF102_2_2_1_128x128_150730-M187474_20151103-144358_deci.h5',
         r'010101JCamF103_2_2_1_128x128_150723-M187201_20151103-144358_deci.h5',
         r'010101JCamF102_2_2_1_128x128_150729-M177929_20151103-144358_deci.h5'
         ]

	fnames_test = [r'20150703JCamF102_4_4_1_128x128_150703-M177929_20160210-100558_deci.h5']
	
	def run_jobs(path, k, base_savepath, base_fname):

		basepath, filename = os.path.split(path)
		root, mousename = os.path.split(basepath)

		savepath = os.path.join(base_savepath, mousename+'-transpose-'+id_tag)
		if os.path.isdir(savepath) == False:
			os.mkdir(savepath)

		save_filename = os.path.join(savepath, filename[:-3]+'-transpose.h5')

		pypath = os.path.join(savepath, base_fname + '.py')
		outpath = os.path.join(savepath, base_fname + '.out')
		txtpath = os.path.join(savepath, base_fname + '.txt')

		file = open(pypath, 'w+') 
		file.writelines( [#'#!/shared/utils.x86_64/python-2.7/bin/python\n' ,
						'import sys\n' ,
						'import gc\n' ,
						'import pickle\n' ,
						'import numpy as np\n' ,
						'sys.path.append(r\'/data/nc-ophys/Matt/\')\n' ,
						'import cats_mapping.widefield_corr_cluster.transpose_hdf5 as td\n' , 
						'\n' ,
						'filepath = \'' + str(path) + '\'\n' , 
						'savepath = \'' + str(save_filename) + '\'\n' , 
						'td.transpose(filepath, savepath)\n'
						] )
		file.close()  
		time.sleep(.2)
		
		jobname = 'Jcorr_batch%d' % k
		file = open(jobname + '.sub', 'w+')
		file.writelines( [ '#!/bin/sh\n',
						 '#PBS -N %s\n' % jobname ,
						 '#PBS -o %s\n' % (outpath) ,
						 '#PBS -q uno\n' ,
						 '#PBS -j oe\n' ,
						 #'#PBS -W x=QOS:high\n' ,
						 '#PBS -l nodes=1:ppn=1\n' ,
						 '#PBS -l walltime=4:00:00\n' ,
						 '#PBS -l vmem=20g\n' ,
						 '#PBS -m a\n' ,
						 '#PBS -r n\n' ,
						 'cd $PBS_O_WORKDIR\n' ,
						 '\n' ,
						 '/shared/utils.x86_64/python-2.7/bin/python ' + pypath + ' > %s\n' % (txtpath)
						 ] )
		file.close()

		os.system('qsub %s' % (jobname + '.sub')) 

		time.sleep(.2)


	''' '''
	''' '''

	project_path = r'/data/nc-ophys/CorticalMapping/IntrinsicImageData'
	#savepath  =r'/data/nc-ophys/Matt/resting state/corrmats'
	savepath = r'/data/nc-ophys/Matt/corrdata/batch_out'

	id_tag = time.strftime("%Y%m%d-%H%M%S") # timestamp identifies the job

	for n,name in enumerate(fnames):    
	    idx = name.index('M')
	    name_id = name[idx-7:idx+7]
	    base_fname = str(name_id) + '_' + str(id_tag)
	    basepath = os.path.join(project_path, name_id)
	    filepath = os.path.join(basepath, name)
	    if os.path.exists(filepath):
			run_jobs(filepath, n, savepath, base_fname)		