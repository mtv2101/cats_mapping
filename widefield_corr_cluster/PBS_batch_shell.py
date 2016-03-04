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
						
	# M187476
	paths = [
	# r'/data/nc-ophys/CorticalMapping/IntrinsicImageData/150722-M187476/150722JCamF103_2_2_1.npy',
	# r'/data/nc-ophys/CorticalMapping/IntrinsicImageData/150723-M187476/010101JCamF102_2_2_1.npy',
	#r'/data/nc-ophys/CorticalMapping/IntrinsicImageData/150727-M187476/150727JCamF102_2_2_1.npy',
	#r'/data/nc-ophys/CorticalMapping/IntrinsicImageData/150729-M187476/010101JCamF102_2_2_1.npy',
	r'/data/nc-ophys/CorticalMapping/IntrinsicImageData/150731-M187476/010101JCamF102_2_2_1.npy',
	
	# M187474
	# r'/data/nc-ophys/CorticalMapping/IntrinsicImageData/150720-M187474/20150720JCamF104'
	# r'/data/nc-ophys/CorticalMapping/IntrinsicImageData/150723-M187474/010101JCamF102_2_2_1.npy',
	#r'/data/nc-ophys/CorticalMapping/IntrinsicImageData/150728-M187474/010101JCamF102_2_2_1.npy',
	#r'/data/nc-ophys/CorticalMapping/IntrinsicImageData/150730-M187474/010101JCamF102_2_2_1.npy',
	r'/data/nc-ophys/CorticalMapping/IntrinsicImageData/150804-M187474/010101JCamF102_2_2_1.npy',
	# r'/data/nc-ophys/CorticalMapping/IntrinsicImageData/150807-M187474/010101JCamF102_2_2_1.npy',
	# r'/data/nc-ophys/CorticalMapping/IntrinsicImageData/150810-M187474/010101JCamF102_2_2_1.npy',
	
	# M187201
	# r'/data/nc-ophys/CorticalMapping/IntrinsicImageData/150720-M187201/20150720JCamF100_2_2_1.npy',
	# r'/data/nc-ophys/CorticalMapping/IntrinsicImageData/150721-M187201/20150721JCamF101'
	# r'/data/nc-ophys/CorticalMapping/IntrinsicImageData/150723-M187201/010101JCamF103_2_2_1.npy',
	#r'/data/nc-ophys/CorticalMapping/IntrinsicImageData/150724-M187201/010101JCamF102_2_2_1.npy',
	#r'/data/nc-ophys/CorticalMapping/IntrinsicImageData/150727-M187201/010101JCamF102_2_2_1.npy',
	# r'/data/nc-ophys/CorticalMapping/IntrinsicImageData/150730-M187201/010101JCamF105_2_2_1.npy',
	# r'/data/nc-ophys/CorticalMapping/IntrinsicImageData/150803-M187201/150803JCamF102_2_2_1.npy',
	# r'/data/nc-ophys/CorticalMapping/IntrinsicImageData/150805-M187201/010101JCamF102_2_2_1.npy',
	r'/data/nc-ophys/CorticalMapping/IntrinsicImageData/150807-M187201/010101JCamF102_2_2_1.npy',
	# r'/data/nc-ophys/CorticalMapping/IntrinsicImageData/150810-M187201/010101JCamF102_2_2_1.npy',
	
	# M177931
	# r'/data/nc-ophys/CorticalMapping/IntrinsicImageData/150701-M177931/20150701JCamF102deci1044.npy',
	# r'/data/nc-ophys/CorticalMapping/IntrinsicImageData/150702-M177931/20150702JCamF104_4_4_1.npy',
	# r'/data/nc-ophys/CorticalMapping/IntrinsicImageData/150703-M177931/20150703JCamF104_4_4_1.npy',
	#r'/data/nc-ophys/CorticalMapping/IntrinsicImageData/150722-M177931/010101JCamF103_2_2_1.npy',
	#r'/data/nc-ophys/CorticalMapping/IntrinsicImageData/150724-M177931/010101JCamF102_2_2_1.npy',
	# r'/data/nc-ophys/CorticalMapping/IntrinsicImageData/150727-M177931/010101JCamF103_2_2_1.npy',
	r'/data/nc-ophys/CorticalMapping/IntrinsicImageData/150729-M177931/010101JCamF102_2_2_1.npy',
	# r'/data/nc-ophys/CorticalMapping/IntrinsicImageData/150731-M177931/010101JCamF102_2_2_1.npy',
	
	# M177929
	# r'/data/nc-ophys/CorticalMapping/IntrinsicImageData/150701-M177929/20150701JCamF101deci1044'
	# r'/data/nc-ophys/CorticalMapping/IntrinsicImageData/150702-M177929/20150702JCamF101_4_4_1.npy',
	# r'/data/nc-ophys/CorticalMapping/IntrinsicImageData/150703-M177929/20150703JCamF102_4_4_1.npy',
	# r'/data/nc-ophys/CorticalMapping/IntrinsicImageData/150704-M177929/20150704JCamF101_4_4_1.npy',
	# r'/data/nc-ophys/CorticalMapping/IntrinsicImageData/150705-M177929/20150705JCamF102_4_4_1.npy',
	# r'/data/nc-ophys/CorticalMapping/IntrinsicImageData/150706-M177929/20150706JCamF101_4_4_1.npy',
	#r'/data/nc-ophys/CorticalMapping/IntrinsicImageData/150722-M177929/150722JCamF102_2_2_1.npy',
	r'/data/nc-ophys/CorticalMapping/IntrinsicImageData/150724-M177929/010101JCamF103_2_2_1.npy',
	# r'/data/nc-ophys/CorticalMapping/IntrinsicImageData/150728-M177929/010101JCamF102_2_2_1.npy',
	# r'/data/nc-ophys/CorticalMapping/IntrinsicImageData/150729-M177929/010101JCamF102_2_2_1.npy',
	# r'/data/nc-ophys/CorticalMapping/IntrinsicImageData/150730-M177929/010101JCamF102_2_2_1.npy',
	# r'/data/nc-ophys/CorticalMapping/IntrinsicImageData/150803-M177929/010101JCamF102_2_2_1.npy'
	]

	path_dict = {r'/data/nc-ophys/CorticalMapping/IntrinsicImageData/150722-M187476/150722JCamF103_2_2_1.npy':
				'150722JCamF103_2_2_1150722-M187476.pkl', 
				r'/data/nc-ophys/CorticalMapping/IntrinsicImageData/150723-M187476/010101JCamF102_2_2_1.npy':
				'010101JCamF102_2_2_1150723-M187476.pkl', 
				r'/data/nc-ophys/CorticalMapping/IntrinsicImageData/150727-M187476/150727JCamF102_2_2_1.npy':
				'150727JCamF102_2_2_1150727-M187476.pkl',
				r'/data/nc-ophys/CorticalMapping/IntrinsicImageData/150729-M187476/010101JCamF102_2_2_1.npy': 
				'010101JCamF102_2_2_1150729-M187476.pkl', 
				r'/data/nc-ophys/CorticalMapping/IntrinsicImageData/150731-M187476/010101JCamF102_2_2_1.npy':
				'010101JCamF102_2_2_1150731-M187476.pkl', 
				r'/data/nc-ophys/CorticalMapping/IntrinsicImageData/150723-M187474/010101JCamF102_2_2_1.npy':
				'010101JCamF102_2_2_1150723-M187474.pkl', 
				r'/data/nc-ophys/CorticalMapping/IntrinsicImageData/150728-M187474/010101JCamF102_2_2_1.npy':
				'010101JCamF102_2_2_1150728-M187474.pkl', 
				r'/data/nc-ophys/CorticalMapping/IntrinsicImageData/150730-M187474/010101JCamF102_2_2_1.npy':
				'010101JCamF102_2_2_1150730-M187474.pkl', 
				r'/data/nc-ophys/CorticalMapping/IntrinsicImageData/150804-M187474/010101JCamF102_2_2_1.npy':
				'010101JCamF102_2_2_1150804-M187474.pkl', 
				r'/data/nc-ophys/CorticalMapping/IntrinsicImageData/150807-M187474/010101JCamF102_2_2_1.npy':
				'010101JCamF102_2_2_1150807-M187474.pkl', 
				r'/data/nc-ophys/CorticalMapping/IntrinsicImageData/150810-M187474/010101JCamF102_2_2_1.npy':
				'010101JCamF102_2_2_1150810-M187474.pkl', 
				r'/data/nc-ophys/CorticalMapping/IntrinsicImageData/150720-M187201/20150720JCamF100_2_2_1.npy':
				'20150720JCamF100_2_2_1150720-M187201.pkl', 
				r'/data/nc-ophys/CorticalMapping/IntrinsicImageData/150723-M187201/010101JCamF103_2_2_1.npy':
				'010101JCamF103_2_2_1150723-M187201.pkl', 
				r'/data/nc-ophys/CorticalMapping/IntrinsicImageData/150723-M187201/010101JCamF103_2_2_1.npy':
				'010101JCamF102_2_2_1150724-M187201.pkl', 
				r'/data/nc-ophys/CorticalMapping/IntrinsicImageData/150727-M187201/010101JCamF102_2_2_1.npy':
				'010101JCamF102_2_2_1150727-M187201.pkl', 
				r'/data/nc-ophys/CorticalMapping/IntrinsicImageData/150730-M187201/010101JCamF105_2_2_1.npy':
				'010101JCamF105_2_2_1150730-M187201.pkl', 
				r'/data/nc-ophys/CorticalMapping/IntrinsicImageData/150803-M187201/150803JCamF102_2_2_1.npy':
				'150803JCamF102_2_2_1150803-M187201.pkl', 
				r'/data/nc-ophys/CorticalMapping/IntrinsicImageData/150805-M187201/010101JCamF102_2_2_1.npy':
				'010101JCamF102_2_2_1150805-M187201.pkl', 
				r'/data/nc-ophys/CorticalMapping/IntrinsicImageData/150807-M187201/010101JCamF102_2_2_1.npy':
				'010101JCamF102_2_2_1150807-M187201.pkl', 
				r'/data/nc-ophys/CorticalMapping/IntrinsicImageData/150810-M187201/010101JCamF102_2_2_1.npy':
				'010101JCamF102_2_2_1150810-M187201.pkl', 
				r'/data/nc-ophys/CorticalMapping/IntrinsicImageData/150722-M177931/010101JCamF103_2_2_1.npy':
				'010101JCamF103_2_2_1150722-M177931.pkl', 
				r'/data/nc-ophys/CorticalMapping/IntrinsicImageData/150724-M177931/010101JCamF102_2_2_1.npy':
				'010101JCamF102_2_2_1150724-M177931.pkl', 
				r'/data/nc-ophys/CorticalMapping/IntrinsicImageData/150727-M177931/010101JCamF103_2_2_1.npy':
				'010101JCamF103_2_2_1150727-M177931.pkl', 
				r'/data/nc-ophys/CorticalMapping/IntrinsicImageData/150729-M177931/010101JCamF102_2_2_1.npy':
				'010101JCamF102_2_2_1150729-M177931.pkl', 
				r'/data/nc-ophys/CorticalMapping/IntrinsicImageData/150731-M177931/010101JCamF102_2_2_1.npy':
				'010101JCamF102_2_2_1150731-M177931.pkl', 
				r'/data/nc-ophys/CorticalMapping/IntrinsicImageData/150722-M177929/150722JCamF102_2_2_1.npy':
				'150722JCamF102_2_2_1150722-M177929.pkl', 
				r'/data/nc-ophys/CorticalMapping/IntrinsicImageData/150724-M177929/010101JCamF103_2_2_1.npy':
				'010101JCamF103_2_2_1150724-M177929.pkl', 
				r'/data/nc-ophys/CorticalMapping/IntrinsicImageData/150729-M177929/010101JCamF102_2_2_1.npy':
				'010101JCamF102_2_2_1150728-M177929.pkl', 
				r'/data/nc-ophys/CorticalMapping/IntrinsicImageData/150729-M177929/010101JCamF102_2_2_1.npy':
				'010101JCamF102_2_2_1150729-M177929.pkl', 
				r'/data/nc-ophys/CorticalMapping/IntrinsicImageData/150730-M177929/010101JCamF102_2_2_1.npy':
				'010101JCamF102_2_2_1150730-M177929.pkl', 
				r'/data/nc-ophys/CorticalMapping/IntrinsicImageData/150803-M177929/010101JCamF102_2_2_1.npy':
				'010101JCamF102_2_2_1150803-M177929.pkl'
				}

	path_dict_h5_128 = {r'/data/nc-ophys/Matt/corrdata/batch_out/20151103-144358_deci/150722JCamF103_2_2_1_128x128_150722-M187476_20151103-144358_deci.h5':
				'150722JCamF103_2_2_1150722-M187476.pkl', 
				r'/data/nc-ophys/Matt/corrdata/batch_out/20151103-144358_deci/010101JCamF102_2_2_1_128x128_150723-M187476_20151103-144358_deci.h5':
				'010101JCamF102_2_2_1150723-M187476.pkl', 
				#r'/data/nc-ophys/Matt/corrdata/batch_out/20151103-144358_deci/150727JCamF102_2_2_1.npy':
				#'150727JCamF102_2_2_1150727-M187476.pkl',
				r'/data/nc-ophys/Matt/corrdata/batch_out/20151103-144358_deci/010101JCamF102_2_2_1_128x128_150729-M187476_20151103-144358_deci.h5': 
				'010101JCamF102_2_2_1150729-M187476.pkl', 
				r'/data/nc-ophys/Matt/corrdata/batch_out/20151103-144358_deci/010101JCamF102_2_2_1_128x128_150731-M187476_20151103-144358_deci.h5':
				'010101JCamF102_2_2_1150731-M187476.pkl', 
				r'/data/nc-ophys/Matt/corrdata/batch_out/20151103-144358_deci/010101JCamF102_2_2_1_128x128_150723-M187474_20151103-144358_deci.h5':
				'010101JCamF102_2_2_1150723-M187474.pkl', 
				r'/data/nc-ophys/Matt/corrdata/batch_out/20151103-144358_deci/010101JCamF102_2_2_1_128x128_150728-M187474_20151103-144358_deci.h5':
				'010101JCamF102_2_2_1150728-M187474.pkl', 
				r'/data/nc-ophys/Matt/corrdata/batch_out/20151103-144358_deci/010101JCamF102_2_2_1_128x128_150730-M187474_20151103-144358_deci.h5':
				'010101JCamF102_2_2_1150730-M187474.pkl', 
				r'/data/nc-ophys/Matt/corrdata/batch_out/20151103-144358_deci/010101JCamF102_2_2_1_128x128_150804-M187474_20151103-144358_deci.h5':
				'010101JCamF102_2_2_1150804-M187474.pkl', 
				r'/data/nc-ophys/Matt/corrdata/batch_out/20151103-144358_deci/010101JCamF102_2_2_1_128x128_150807-M187474_20151103-144358_deci.h5':
				'010101JCamF102_2_2_1150807-M187474.pkl', 
				r'/data/nc-ophys/Matt/corrdata/batch_out/20151103-144358_deci/010101JCamF102_2_2_1_128x128_150810-M187474_20151103-144358_deci.h5':
				'010101JCamF102_2_2_1150810-M187474.pkl', 
				#r'/data/nc-ophys/Matt/corrdata/batch_out/20151103-144358_deci/20150720JCamF100_2_2_1.npy':
				#'20150720JCamF100_2_2_1150720-M187201.pkl', 
				r'/data/nc-ophys/Matt/corrdata/batch_out/20151103-144358_deci/010101JCamF103_2_2_1_128x128_150723-M187201_20151103-144358_deci.h5':
				'010101JCamF103_2_2_1150723-M187201.pkl', 
				r'/data/nc-ophys/Matt/corrdata/batch_out/20151103-144358_deci/010101JCamF102_2_2_1_128x128_150724-M187201_20151103-144358_deci.h5':
				'010101JCamF102_2_2_1150724-M187201.pkl', 
				r'/data/nc-ophys/Matt/corrdata/batch_out/20151103-144358_deci/010101JCamF102_2_2_1_128x128_150727-M187201_20151103-144358_deci.h5':
				'010101JCamF102_2_2_1150727-M187201.pkl', 
				r'/data/nc-ophys/Matt/corrdata/batch_out/20151103-144358_deci/010101JCamF105_2_2_1_128x128_150730-M187201_20151103-144358_deci.h5':
				'010101JCamF105_2_2_1150730-M187201.pkl', 
				#r'/data/nc-ophys/Matt/corrdata/batch_out/20151103-144358_deci/150803JCamF102_2_2_1.npy':
				#'150803JCamF102_2_2_1150803-M187201.pkl', 
				r'/data/nc-ophys/Matt/corrdata/batch_out/20151103-144358_deci/010101JCamF102_2_2_1_128x128_150805-M187201_20151103-144358_deci.h5':
				'010101JCamF102_2_2_1150805-M187201.pkl', 
				r'/data/nc-ophys/Matt/corrdata/batch_out/20151103-144358_deci/010101JCamF102_2_2_1_128x128_150807-M187201_20151103-144358_deci.h5':
				'010101JCamF102_2_2_1150807-M187201.pkl', 
				r'/data/nc-ophys/Matt/corrdata/batch_out/20151103-144358_deci/010101JCamF102_2_2_1_128x128_150810-M187201_20151103-144358_deci.h5':
				'010101JCamF102_2_2_1150810-M187201.pkl', 
				r'/data/nc-ophys/Matt/corrdata/batch_out/20151103-144358_deci/010101JCamF103_2_2_1_128x128_150722-M177931_20151103-144358_deci.h5':
				'010101JCamF103_2_2_1150722-M177931.pkl', 
				r'/data/nc-ophys/Matt/corrdata/batch_out/20151103-144358_deci/010101JCamF102_2_2_1_128x128_150724-M177931_20151103-144358_deci.h5':
				'010101JCamF102_2_2_1150724-M177931.pkl', 
				r'/data/nc-ophys/Matt/corrdata/batch_out/20151103-144358_deci/010101JCamF103_2_2_1_128x128_150727-M177931_20151103-144358_deci.h5':
				'010101JCamF103_2_2_1150727-M177931.pkl', 
				r'/data/nc-ophys/Matt/corrdata/batch_out/20151103-144358_deci/010101JCamF102_2_2_1_128x128_150729-M177931_20151103-144358_deci.h5':
				'010101JCamF102_2_2_1150729-M177931.pkl', 
				r'/data/nc-ophys/Matt/corrdata/batch_out/20151103-144358_deci/010101JCamF102_2_2_1_128x128_150731-M177931_20151103-144358_deci.h5':
				'010101JCamF102_2_2_1150731-M177931.pkl', 
				r'/data/nc-ophys/Matt/corrdata/batch_out/20151103-144358_deci/150722JCamF102_2_2_1_128x128_150722-M177929_20151103-144358_deci.h5':
				'150722JCamF102_2_2_1150722-M177929.pkl', 
				r'/data/nc-ophys/Matt/corrdata/batch_out/20151103-144358_deci/010101JCamF103_2_2_1_128x128_150724-M177929_20151103-144358_deci.h5':
				'010101JCamF103_2_2_1150724-M177929.pkl', 
				r'/data/nc-ophys/Matt/corrdata/batch_out/20151103-144358_deci/010101JCamF102_2_2_1_128x128_150728-M177929_20151103-144358_deci.h5':
				'010101JCamF102_2_2_1150728-M177929.pkl', 
				r'/data/nc-ophys/Matt/corrdata/batch_out/20151103-144358_deci/010101JCamF102_2_2_1_128x128_150729-M177929_20151103-144358_deci.h5':
				'010101JCamF102_2_2_1150729-M177929.pkl', 
				r'/data/nc-ophys/Matt/corrdata/batch_out/20151103-144358_deci/010101JCamF102_2_2_1_128x128_150730-M177929_20151103-144358_deci.h5':
				'010101JCamF102_2_2_1150730-M177929.pkl', 
				r'/data/nc-ophys/Matt/corrdata/batch_out/20151103-144358_deci/010101JCamF102_2_2_1_128x128_150803-M177929_20151103-144358_deci.h5':
				'010101JCamF102_2_2_1150803-M177929.pkl'
				}

	path_dict_h5_256 = {r'/data/nc-ophys/Matt/corrdata/batch_out/20151117-224455_deci/010101JCamF102_2_2_1_256x256_150724-M187201_20151117-224455_deci.h5':
				'010101JCamF102_2_2_1150724-M187201.pkl',
				r'/data/nc-ophys/Matt/corrdata/batch_out/20151117-224455_deci/010101JCamF102_2_2_1_256x256_150729-M177931_20151117-224455_deci.h5':
				'010101JCamF102_2_2_1150729-M177931.pkl',
				r'/data/nc-ophys/Matt/corrdata/batch_out/20151117-224455_deci/010101JCamF102_2_2_1_256x256_150804-M187474_20151117-224455_deci.h5':
				'010101JCamF102_2_2_1150804-M187474.pkl', 
				r'/data/nc-ophys/Matt/corrdata/batch_out/20151117-224455_deci/150722JCamF102_2_2_1_256x256_150722-M177929_20151117-224455_deci.h5':
				'150722JCamF102_2_2_1150722-M177929.pkl',
				r'/data/nc-ophys/Matt/corrdata/batch_out/20151117-224455_deci/150727JCamF102_2_2_1_256x256_150727-M187476_20151117-224455_deci.h5':
				'150727JCamF102_2_2_1150727-M187476.pkl',
	}
	
	test_path_dict = {r'/data2/nc-ophys/Matt/corrdata/batch_out/20151103-144358_deci/010101JCamF102_2_2_1_128x128_150729-M177931_20151103-144358_deci.h5':
				'010101JCamF102_2_2_1150729-M177931.pkl'}  

	doug_test_paths - ["/data/nc-ophys/CorticalMapping/IntrinsicImageData/150805-M187478/010101JCamF102_16_16_1.npy",
        "data/nc-ophys/CorticalMapping/IntrinsicImageData/150804-M187474/010101JCamF102_16_16_1.npy"]
	
	def run_jobs(path, k, base_savepath, pkl):

		#print path, k, savepath, pkl

		id_tag = time.strftime("%Y%m%d-%H%M%S")
		basepath, filename = os.path.split(path)
		root, mousename = os.path.split(basepath)
		save_root, savename = os.path.split(base_savepath)
		#savepath = os.path.join(base_savepath, savename) # give the .out, .txt, .py files same name as parent folder
		savepath = os.path.join(base_savepath, filename[:-3])
		shell_fpath = os.path.join(base_savepath, filename[:-3]) + '.py'

		pklpath = r'/data/nc-ophys/Matt/corrdata/batch_out/20151114_batch_parse_activity'
		idx_path = os.path.join(pklpath, pkl)
		
		file = open(shell_fpath, 'w+') 
		file.writelines( [#'#!/shared/utils.x86_64/python-2.7/bin/python\n' ,
						'import sys\n' ,
						'import gc\n' ,
						'import pickle\n' ,
						'sys.path.append(r\'/data/nc-ophys/Matt/\')\n' ,
						'import aibs.CorticalMapping.widefield_corr_cluster.Jcorr as jc\n' , 
						#'import aibs.CorticalMapping.widefield_corr_cluster.bilateral_correlation_centroid_clustering as cc\n' ,
						'filepath = \'' + str(path) + '\'\n' , 
						'savepath = \'' + str(base_savepath) + '\'\n' , 
						'indices = pickle.load(open(\'' + str(idx_path) + '\', \'rU\'))\n' ,
						'for m in range(2):\n', 
						'\tcorr = jc.Jcorr(filepath,savepath,100,1,frameNum=indices[m],do_bandpass=False,moving_dff=True,SAVE_DFF=False,img_source = \'h5\',do_corr=True,window=60,special_string=\'_\'+\''+str(mousename)+'\'+\'_\'+str(m))\n' , 
						'\tcorr.return_corr(corr)\n' ,  
						#'corr_cluster = cc.ccc(corr.savepath, corr.corrsavepath, corr.maskpath, corr_thresh=.8, eps=1, min_samples=100)\n' ,
						#'corr_cluster.plot_all(corr_cluster.all_centroids, corr_cluster.class_centroids, corr_cluster.areas, corr_cluster.seed_pixel_maps)'                      
						'del corr\n' ,
						'gc.collect()'
						] )
		file.close()  
		time.sleep(2.)
		
		jobname = 'Jcorr_batch%d' % k
		file = open(jobname + '.sub', 'w+')
		file.writelines( [ '#!/bin/sh\n',
						 '#PBS -N %s\n' % jobname ,
						 '#PBS -o %s\n' % (savepath + '.out') ,
						 '#PBS -q mindscope\n' ,
						 '#PBS -j oe\n' ,
						 '#PBS -l nodes=1:ppn=1\n' ,
						 '#PBS -l walltime=200:00:00\n' ,
						 '#PBS -l vmem=120g\n' ,
						 '#PBS -m a\n' ,
						 '#PBS -r n\n' ,
						 'cd $PBS_O_WORKDIR\n' ,
						 '\n' ,
						 '/shared/utils.x86_64/python-2.7/bin/python ' + shell_fpath + ' > %s\n' % (savepath + '.txt')
						 ] ) 
		file.close()

		os.system('qsub %s' % (jobname + '.sub')) 

		time.sleep(2.)


	''' '''
	''' '''

	#batch_path = r'/data/nc-ophys/Matt/corrdata/batch_out/20151103-144358_deci'
	#batch_path = r'/data/nc-ophys/Matt/corrdata/batch_out/test'
	#batch_path = r'\\aibsdata2\nc-ophys\Matt\corrdata\batch_out\20151103-144358_deci'

	# def is_data(name, path):
	# 	namepath = os.path.join(path, name)
	# 	if os.path.isfile(namepath):
	# 		if fnmatch.fnmatch(name, '*.h5'):
	# 			return namepath

	# batch_paths = [is_data(name, batch_path) for name in os.listdir(batch_path) if is_data(name, batch_path)]

	id_tag = time.strftime("%Y%m%d-%H%M%S") + '_batch_corr'
	savepath = r'/data/nc-ophys/Matt/corrdata/batch_out'
	savefolder = os.path.join(savepath, id_tag)
	os.mkdir(savefolder)

	#[run_jobs(path, k, savefolder) for k,path in enumerate(batch_paths)]
	#[run_jobs(path, k, savefolder, path_dict[path]) for k,path in enumerate(path_dict.keys())]

	[run_jobs(path, k, savefolder, doug_test_paths[path]) for k,path in enumerate(doug_test_paths.keys())]

		