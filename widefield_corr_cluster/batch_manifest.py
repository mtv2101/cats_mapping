#!/shared/utils.x86_64/python-2.7/bin/python
"""
Created on Thu Oct 29 10:08:50 2015

@author: mattv
"""

import os
import sys
import platform
import json
import socket
import string
if platform.system() == 'Linux': 
    sys.path.append(r'/data/nc-ophys/Matt/imagingbehavior')
    sys.path.append(r'/data/nc-ophys/Matt')
    import imaging_behavior as ib
elif platform.system() == 'Windows':
    import imaging_behavior as ib
import fnmatch
import pdb
import re


manifest_keys = [ 
                 'jcam_movie_file_path',
                 'jphys_data_file_path',
                 'behavioral_log_file_path',
                 ]

def create_manifest(save_file_name=None, initial_session_dir='./'):
    
    session_description_dict = {}
    initialdir=initial_session_dir
    
    # get JCam .h5 file above 100MB
    jcamf = re.compile(r'.*JCam.*')
    jcam_full_list = [file_name for file_name in os.listdir(initialdir) if jcamf.match(file_name)]
    jcam_list = [file_name for file_name in jcam_full_list if os.path.getsize(os.path.join(initialdir, file_name)) > 100000000 and fnmatch.fnmatch(file_name, '*.h5')]
    if len(jcam_list) == 1:
        session_description_dict['jcam_movie_file_path'] = os.path.join(initialdir, jcam_list[0])        
        file_tag = session_description_dict['jcam_movie_file_path'].split('JCam')[-1]
        file_tag = file_tag[1:4]
    else:
        print jcam_list
        session_description_dict['jcam_movie_file_path'] = ib.io.ask_file_tk(title='Select file for JCam', initialdir=initialdir)
        file_tag = session_description_dict['jcam_movie_file_path'].split('JCam')[-1]
        file_tag = file_tag[1:4]
        
    # Get JPhys file using filetag:    
    jphysf = re.compile(r'.*JPhys%s.*' % file_tag)
    jphysf_list = [file_name for file_name in os.listdir(initialdir) if jphysf.match(file_name)]
    assert len(jphysf_list) <= 1
    if len(jphysf_list) == 1:
        session_description_dict['jphys_data_file_path'] = os.path.join(initialdir, jphysf_list[0])        
    else:
        print jphysf_list
        session_description_dict['jphys_data_file_path'] = ib.io.ask_file_tk(title='Select file for JPhys', initialdir=initialdir)
        
    # Get behavioral log pickle file in current directory:
    r = re.compile(r'.*.pkl')
    pickle_list = [file_name for file_name in os.listdir(initialdir) if r.match(file_name)]
    if len(pickle_list) == 1:
        session_description_dict['behavioral_log_file_path'] = os.path.join(initialdir, pickle_list[0])  
    else:
        print pickle_list
        session_description_dict['behavioral_log_file_path'] = ib.io.ask_file_tk(title='Select file for pkl', initialdir=initialdir)
    
    write_session_manifest(os.path.join(initialdir, save_file_name), session_description_dict)

def write_session_manifest(save_file_name, manifest_dict):
    
    dump_dict = {}
    for key, absolute_file_name_list in manifest_dict.items():
        if absolute_file_name_list == None:
            dump_dict[key] = None
        else:
            assert key in manifest_keys
            assert os.path.exists(absolute_file_name_list)
            dump_dict[key] = absolute_file_name_list
        
    with open(os.path.join(save_file_name), 'w') as outfile:
        json.dump(dump_dict, outfile, indent=2)
    
    print str(save_file_name) + ' saved sucessfully!\n'
        
        
if __name__ == "__main__":
            
    fpaths = 'Z:\DropBox\Cortical Dynamics\CATS 128x128'
    fnames = os.listdir(fpaths)

    destpath = r'\\aibsdata2\nc-ophys\CorticalMapping\IntrinsicImageData'

    #fnames = [r'\\aibsdata2\nc-ophys\CorticalMapping\IntrinsicImageData\150728-M177929']
    #fnames = ['010101JCamF102_2_2_1_128x128_150728-M177929_20151103-144358_deci.h5']

    for n,name in enumerate(fnames):    
        idx = name.index('M')
        name_id = name[idx-7:idx+7]
        print 'building ' + str(name_id)
        savepath = os.path.join(destpath, name_id)
        filepath = os.path.join(savepath, name)
        if os.path.exists(filepath):
            manifest_id = name_id + '_128x128h5_manifest.json'
            manifest_name = os.path.join(savepath, manifest_id)
            #print manifest_name
            create_manifest(manifest_name, savepath) 
        else:
            print str(name_id) + ' not found'