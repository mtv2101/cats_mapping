# -*- coding: utf-8 -*-
"""
Created on Fri Mar 04 14:55:44 2016

@author: chrism
"""
from imp import load_dynamic  

SHARED_CV2_DLL_PATH = r"\\allen\programs\braintv\workgroups\nc-ophys\shared_cv2\cv2.pyd"

def import_from_dll(dll_path=SHARED_CV2_DLL_PATH):
    return load_dynamic("cv2", dll_path)

CV2_DLL = import_from_dll()