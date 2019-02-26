# -*- coding: utf-8 -*-
"""
Created on Wed Jan 06 15:14:10 2016

@author: chrism
"""

import numpy as np
from copy import deepcopy
from warnings import warn

from alignmentFunctions import *
from alignmentFileTools import *

def test_affine_sid_eroded(mapping_arr_uint8,exp_arr_uint8,
                           transformation_matrix=None,cats=False):
    max_shape = max(mapping_arr_uint8.shape[0],exp_arr_uint8.shape[0])
    if max_shape == 2048:
        replace_kernel = 55
    elif max_shape == 1024:
        replace_kernel = 25
    else:
        replace_kernel = None
    if cats:
        replace_kernel /= 2
    if not isinstance(transformation_matrix,np.ndarray):
        aligned_mapping_arr_uint8 = mapping_arr_uint8
    else:
        aligned_mapping_arr_uint8 = apply_affine_transform(mapping_arr_uint8,
                                                           exp_arr_uint8,
                                                           transformation_matrix)
    
    return calc_thresholded_eroded_sid(aligned_mapping_arr_uint8,
                                       exp_arr_uint8,
                                       replace_kernel=replace_kernel)

def automagic_registration_projective(mapping_img,exp_img,detector="SURF"):
    mapping_arr_uint8 = convert_array_to_norm(mapping_img,dtype=np.uint8)
    exp_arr_uint8 = convert_array_to_norm(exp_img,dtype=np.uint8)
    target_adapt = adaptive_histogram_equalization(mapping_arr_uint8)
    reference_adapt = adaptive_histogram_equalization(exp_arr_uint8)
    good_matches,kp0,kp1 = match_key_points(target_adapt,reference_adapt,match_thresh=0.7,
                                            detector=detector)
    projective_matrix,mask = get_projective_transform(good_matches,kp0,kp1)
    transform_img = get_transform_image(deepcopy(target_adapt),
                                        deepcopy(reference_adapt),
                                        kp0,kp1,projective_matrix,mask,
                                        good_matches)
    registered_img = apply_perspective_transform(mapping_img,
                                                 exp_img,projective_matrix)
    
    return (mapping_arr_uint8,exp_arr_uint8,target_adapt,reference_adapt,
            good_matches,kp0,kp1,projective_matrix,mask,transform_img,
            registered_img)

def write_affine_to_json(affine_transform,json_path):
    import os
    import json
    
    if isinstance(affine_transform,np.ndarray):
        affine_transform = affine_transform.tolist()
        
    directory_name = os.path.dirname(json_path)
    if not os.path.exists(directory_name):
        os.makedirs(directory_name)
    
    with open(json_path,"w") as out:
        json.dump(affine_transform,out,indent=4)

def load_affine_transform_from_json(json_path):
    import json
    
    with open(json_path) as f:
        affine_transform = json.load(f)
        
    return np.array(affine_transform)

def automatic_image_registration(mapping_img,exp_img,*args,**kwargs):
    
    ret = automagic_registration_affine_persistent_bf_scored_detector_matcher(mapping_img,
                                                                              exp_img,
                                                                              *args,**kwargs)
    mapping_arr_uint8,exp_arr_uint8,target_adapt,reference_adapt = ret[:4]
    good_matches,kp0,kp1,affine_matrix,transform_img,registered_img = ret[4:]
     
    abs_diff_img = cv2.absdiff(convert_array_to_norm(registered_img),
                               convert_array_to_norm(exp_img))
    return affine_matrix,transform_img,abs_diff_img
    
def automagic_registration_affine_persistent_bf_scored_detector_matcher(mapping_img,exp_img,
                                                                        threshold_score=20000,
                                                                        cats=False,
                                                                        match_threshs=(0.7,0.8,0.85),
                                                                        adaptive_clip=2.0,
                                                                        adaptive_tile_dim=8,
                                                                        detectors=("SIFT","SURF"),
                                                                        matchers=("brute","flann")):
    registration_kwargs = {
                        "threshold_score": threshold_score,
                        "match_threshs": match_threshs,
                        "adaptive_clip": adaptive_clip,
                        "adaptive_tile_dim": adaptive_tile_dim,
                        "cats": cats
    }
    
    for matcher in matchers:
        for detector in detectors:
            try:
                results = automagic_registration_affine_persistent_bf_scored(mapping_img,exp_img,
                                                                             detector=detector,
                                                                             matcher=matcher,
                                                                             **registration_kwargs)
                return results
            except Exception as e:
                print e
    else:
        registration_kwargs.pop("threshold_score")
        registration_kwargs.pop("match_threshs")
        registration_kwargs.pop("cats")
        for matcher in matchers:
            for detector in detectors:
                for match_thresh in match_threshs:
                    try:
                        
                        ret = automagic_registration_affine_persistent_bf(mapping_img,exp_img,
                                                                      detector=detector,
                                                                      matcher=matcher,
                                                                      match_thresh=match_thresh,
                                                                      **registration_kwargs)
                        return ret
                    except Exception as e:
                        print e
        else:
            print "Couldn't even use non-scored version!"
            return automagic_registration_affine_persistent_bf(mapping_img,exp_img,
                                                               match_thresh=0.7,
                                                               adaptive_clip=2.0,
                                                               adaptive_tile_dim=8,
                                                               detector="SURF",
                                                               matcher="flann")
    

def automagic_registration_affine_persistent_bf_scored(mapping_img,exp_img,
                                                       threshold_score=25000,
                                                       cats=False,
                                                       match_threshs=(0.7,0.8,0.85),
                                                       adaptive_clip=2.0,
                                                       adaptive_tile_dim=8,
                                                       detector="SIFT",
                                                       matcher="brute"):
    
    registration_kwargs = {
                        "adaptive_clip": adaptive_clip,
                        "adaptive_tile_dim": adaptive_tile_dim,
                        "detector": detector,
                        "matcher": matcher,
    }
    
    for match_thresh in match_threshs:
        results = automagic_registration_affine_persistent_bf(mapping_img,exp_img,
                                                              match_thresh=match_thresh,
                                                              **registration_kwargs)
        mapping_arr_uint8 = results[0]
        exp_arr_uint8 = results[1]
        affine_matrix = results[-3]
        sid_eroded = test_affine_sid_eroded(mapping_arr_uint8,exp_arr_uint8,
                                            affine_matrix,cats=cats)
        if sid_eroded < threshold_score:
            break
    else:
        raise Exception("Could get a good score for any match_thresh.")
    
    return results

def automagic_registration_affine_persistent_bf(mapping_img,exp_img,
                                                match_thresh=0.7,
                                                adaptive_clip=2.0,
                                                adaptive_tile_dim=8,
                                                detector="SIFT",
                                                matcher="flann"):
    results = persistent_affine_registration_bf(mapping_img,exp_img,
                                                n_erosions=0,
                                                detector=detector,
                                                matcher=matcher,
                                                match_thresh=match_thresh,
                                                clip_range=(adaptive_clip,),
                                                tile_range=(adaptive_tile_dim,))
    if not results:
        results = persistent_affine_registration_bf(mapping_img,exp_img,
                                                    n_erosions=1,
                                                    detector=detector,
                                                    matcher=matcher,
                                                    match_thresh=match_thresh)
        print detector,matcher,match_thresh

        if not results:
            raise Exception("registration failed")
    
    return results

def persistent_affine_registration_bf(mapping_img,exp_img,n_erosions,
                                      detector="SIFT",
                                      matcher="flann",
                                      clip_range=range(2,12,2),
                                      tile_range=(8,16,32,64,80),
                                      match_thresh=0.7):
    for clip,tile in zip(clip_range,tile_range):
        try:
            eroded_mapping_img = box_enhanced_binary_erosion(mapping_img,
                                                             iterations=n_erosions)
            eroded_exp_img = box_enhanced_binary_erosion(exp_img,
                                                         iterations=n_erosions)
            ret = automagic_registration_affine(eroded_mapping_img,
                                                eroded_exp_img,
                                                match_thresh=match_thresh,
                                                adaptive_clip=clip,
                                                adaptive_tile_dim=tile,
                                                detector=detector,
                                                matcher=matcher)
            break
        except Exception as e:
            warn(str(e))
    else:
        ret = None
    return ret #last idx is all you care about

def automagic_registration_affine(mapping_img,exp_img,match_thresh=0.7,
                                  adaptive_clip=2.0,adaptive_tile_dim=8,
                                  detector="SIFT",matcher="flann"):
    mapping_arr_uint8 = convert_array_to_norm(mapping_img,dtype=np.uint8)
    exp_arr_uint8 = convert_array_to_norm(exp_img,dtype=np.uint8)
    
    adaptive_args = (adaptive_clip,(adaptive_tile_dim,adaptive_tile_dim))
    target_adapt = adaptive_histogram_equalization(mapping_arr_uint8,
                                                   *adaptive_args)
        
    reference_adapt = adaptive_histogram_equalization(exp_arr_uint8,
                                                      *adaptive_args)
                                                
    good_matches,kp0,kp1 = match_key_points(target_adapt,
                                            reference_adapt,
                                            match_thresh=match_thresh,
                                            detector=detector,
                                            matcher=matcher)
    
    affine_matrix = get_affine_transform(good_matches,kp0,kp1)
    
    transform_img = get_transform_image(deepcopy(target_adapt),
                                        deepcopy(reference_adapt),
                                        kp0,kp1,affine_matrix,None,
                                        good_matches,affine=True)
                                      
    registered_img = apply_affine_transform(mapping_img,exp_img,affine_matrix)

    return (mapping_arr_uint8,exp_arr_uint8,target_adapt,reference_adapt,
            good_matches,kp0,kp1,affine_matrix,transform_img,registered_img)

if __name__ == "__main__":
    MAPPING_IMG_PATH = r"\\aibsdata2\nc-ophys\CorticalMapping\IntrinsicImageData\ProcessedMaps\2015.01.16-M156569\20150116_M156569_Trial1_2_3_4.pkl"
    EXP_IMG_050715 = r"\\AIBSDATa2\nc-ophys\ImageData\Doug\150507-M156569\150507JCamF100"
    exp_img = load_experiment_img_jcamf(EXP_IMG_050715, {"row":2048, "column":2048})
    mapping_img = load_mapping_img(MAPPING_IMG_PATH)
    ret = automatic_image_registration(mapping_img, exp_img)
    write_affine_to_json(ret[0],r"C:\Users\chrism\Desktop\alignment_tests\test.json")