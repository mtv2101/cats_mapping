# -*- coding: utf-8 -*-
"""
Created on Thu Nov 12 20:53:37 2015

@author: chrism
"""
import numpy as np
from import_shared_cv2 import CV2_DLL as cv2
#cv2 = cv2v3.cv2

from alignmentFileTools import convert_array_to_norm

class RootSIFT:
    def __init__(self):
        # initialize the SIFT feature extractor
        self.extractor = cv2.DescriptorExtractor_create("SIFT")

    def compute(self, image, kps, eps=1e-7):
        # compute SIFT descriptors
        (kps, descs) = self.extractor.compute(image, kps)

        # if there are no keypoints or descriptors, return an empty tuple
        if len(kps) == 0:
            return ([], None)

        # apply the Hellinger kernel by first L1-normalizing and taking the
        # square-root
        descs /= (descs.sum(axis=1, keepdims=True) + eps)
        descs = np.sqrt(descs)
        #descs /= (np.linalg.norm(descs, axis=1, ord=2) + eps)

        # return a tuple of the keypoints and descriptors
        return (kps, descs)

def adaptive_histogram_equalization(img,clipLimit=5.0,tileGridSize=(20,20)):
    clahe = cv2.createCLAHE(clipLimit=clipLimit,tileGridSize=tileGridSize)
    return clahe.apply(img)

def match_key_points(img0,img1,match_thresh=0.7,detector="SURF",matcher="flann"):
    FLANN_INDEX_KDTREE = 0    
    
    #initialize surf detector
    if detector == "SIFT":
        det = cv2.xfeatures2d.SIFT_create()
    else:
        det = cv2.xfeatures2d.SURF_create() 
    
    #intitialize flann matcher or brute matcher
    if matcher.lower() == "brute":
        matcher = cv2.BFMatcher(cv2.NORM_L1,crossCheck=False)
    else:
        idx_params = dict(algorithm=FLANN_INDEX_KDTREE,trees=5)
        srch_params = dict(checks=50)
        matcher = cv2.FlannBasedMatcher(idx_params,srch_params)

    print detector
    
    if detector == "SIFT":
        root = RootSIFT()
        kp0 = det.detect(img0)
        kp1 = det.detect(img1)
        kp0,des0 = root.compute(img0,kp0)
        kp1,des1 = root.compute(img1,kp1)
    else:
        kp0,des0 = det.detectAndCompute(img0,None)
        kp1,des1 = det.detectAndCompute(img1,None)
    
    matches = matcher.knnMatch(des0,des1,k=2)
    
    #determine good matches with match_thresh
    good_matches = [m for m,n in matches if (m.distance < match_thresh*n.distance)]
    
    return good_matches,kp0,kp1

def get_affine_transform(matches,kp0,kp1,full_affine=False):
    src_pts = np.float32([kp0[m.queryIdx].pt for m in matches]).reshape(-1,1,2)
    dst_pts = np.float32([kp1[m.trainIdx].pt for m in matches]).reshape(-1,1,2)
    
    return cv2.estimateRigidTransform(src_pts,dst_pts,fullAffine=full_affine)

def get_affine_transform_persistent(good_matches,kp0,kp1,full_affine=False):

    affine_matrix = get_affine_transform(good_matches,kp0,kp1,full_affine)
    
    if affine_matrix == None:
        for idx in range(len(list(good_matches))):
            reduced_matches = list(good_matches)
            reduced_matches.pop(idx)
            affine_matrix = get_affine_transform(reduced_matches,kp0,kp1,
                                                 full_affine)
            if affine_matrix != None:
                break
    else:
        reduced_matches = good_matches
        
    return affine_matrix,reduced_matches

def box_enhanced_binary_erosion(img,erosion_kernel_size=(3,3),
                                box_kernel_size=(5,5),
                                box_kernel_depth=-1,
                                iterations=1):
    ret_img = np.array(img)
    erosion_kernel = np.ones(erosion_kernel_size,np.uint8) #binary erosion kernel
    box_kwargs = {"ksize":box_kernel_size,"ddepth":box_kernel_depth}
    for i in range(iterations):
        ret_img = cv2.erode(cv2.boxFilter(ret_img,**box_kwargs),erosion_kernel)
    return ret_img

def get_projective_transform(matches,kp0,kp1,**kwargs):
    homography_kwargs = {
                        "method": cv2.RANSAC,
                        "ransacReprojThreshold": 5.0
    }
    homography_kwargs.update(kwargs)
    
    src_pts = np.float32([kp0[m.queryIdx].pt for m in matches]).reshape(-1,1,2)
    dst_pts = np.float32([kp1[m.trainIdx].pt for m in matches]).reshape(-1,1,2)
    
    return cv2.findHomography(src_pts,dst_pts,**homography_kwargs)
    
def get_transform_image(img0,img1,kp0,kp1,transform_matrix,mask,matches,
                        affine=False):
    if isinstance(mask,np.ndarray):
        matches_mask = mask.ravel().tolist()
    else:
        matches_mask = None
    
    h,w = img0.shape
    pts = np.float32([[0,0],[0,h-1],[w-1,h-1],[w-1,0]]).reshape(-1,1,2)
    if affine:
        dst = cv2.transform(pts,transform_matrix) #this seemed to work well...todo verify this does what you think it does...
    else:
        dst = cv2.perspectiveTransform(pts,transform_matrix)

    img2 = cv2.polylines(img1,[np.int32(dst)],True,255,3, cv2.LINE_AA)

    draw_params = dict(singlePointColor=None,
                       matchesMask=matches_mask, # draw only inliers
                       flags=2)

    return cv2.drawMatches(img0,kp0,img2,kp1,matches,None,**draw_params)

def apply_affine_transform(target_image,reference_image,transform_matrix,
                           borderValue=None):
    if not borderValue:
        borderValue = int(np.amin(reference_image))
        if borderValue > 100:
            borderValue = int(np.amin(convert_array_to_norm(reference_image)))
    
    return cv2.warpAffine(target_image,
                               transform_matrix,
                               reference_image.shape,
                               borderValue=borderValue)

def apply_perspective_transform(target_image,reference_image,transform_matrix,
                                borderValue=None):
    if not borderValue:
        borderValue = int(np.amin(reference_image))
    
    return cv2.warpPerspective(target_image,
                               transform_matrix,
                               reference_image.shape,
                               borderValue=borderValue)

def modify_projective_transform_phase(transform_matrix,modifications=(0,0)):
    """
    Expects  at least a 2x3 matrix
    """
    new_transform_matrix = np.array(transform_matrix)
    new_transform_matrix[0][2] += modifications[0]
    new_transform_matrix[1][2] += modifications[1]
    return new_transform_matrix

def get_projective_transform_phase(transform_matrix):
    return transform_matrix[0][2],transform_matrix[1][2]

#def modify_projective_transform_scale_rot(transform_matrix,
#                                           modifications=(0,0,0)):
#    """
#    this gives a good idea of what I'm trying to do here:
#    http://math.stackexchange.com/questions/13150/extracting-rotation-scale-values-from-2d-transformation-matrix
#    """
#    new_transform_matrix = np.array(transform_matrix)    
#    
#    scale_x,scale_y,rot_theta = get_projective_transform_scale_rot(new_transform_matrix)
#    scale_x += modifications[0]
#    scale_y += modifications[1]
#    rot_theta += modifications[2]
    
def get_projective_transform_scale_rot(transform_matrix):
    transform_matrix = np.array(transform_matrix) #just in case
    a = transform_matrix[0][0]
    b = transform_matrix[0][1]
    c = transform_matrix[1][0]
    d = transform_matrix[1][1]
    scale_x = np.sign(a)*np.sqrt((a**2)+(b**2))
    scale_y = np.sign(d)*np.sqrt((c**2)+(d**2))
    rot_theta_0 = np.arctan2(-b,a)
    rot_theta_1 = np.arctan2(c,d)
    if rot_theta_0 != rot_theta_1:
        raise ValueError("Both rot_thetas need to be equal...")
    return scale_x,scale_y,rot_theta_0
    
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from copy import deepcopy
    
    img0 =  cv2.imread(r"\\AIBSDATA2\nc-ophys\1022\These are images\match_img10.png",0)
    img1 =  cv2.imread(r"\\AIBSDATA2\nc-ophys\1022\These are images\ref_img10.png",0)
    
    img0_adapt = adaptive_histogram_equalization(img0)
    img1_adapt = adaptive_histogram_equalization(img1)
    
    img0_adapt = cv2.imread(r"C:\Users\chrism\Desktop\Images\adaptive_match_ext.png",0)
    img1_adapt = cv2.imread(r"C:\Users\chrism\Desktop\Images\adaptive_ref_ext.png",0)
    
    good_matches,kp0,kp1 = match_key_points(img0_adapt,img1_adapt)
    
    projective_matrix,mask = get_projective_transform(good_matches,kp0,kp1)
    
    transform_img = get_transform_image(deepcopy(img0_adapt),deepcopy(img1_adapt),
                                        kp0,kp1,projective_matrix,mask,
                                        good_matches)
        
    #applies a perspective transformation using a projective transformation matrix!                                  
    registered_img = apply_perspective_transform(img0,img1, 
                                                 projective_matrix)
    
    fig,ax = plt.subplots(1,2)
    fig.suptitle = "adaptive histogram results"
    ax[0].imshow(img0_adapt,"gray")
    ax[0].set_title("target")
    ax[1].imshow(img1_adapt,"gray")
    ax[1].set_title("reference")
    plt.show()
    
    fig,ax = plt.subplots(1,1)
    ax.imshow(transform_img)
    plt.show()
    
    fig,ax = plt.subplots(1,2)
    fig.suptitle = "registration results"
    ax[0].imshow(registered_img,"gray")
    ax[0].set_title("registered")
    ax[1].imshow(img1,"gray")
    ax[1].set_title("reference")
    plt.show()
    
    plt.imshow(cv2.absdiff(registered_img,img1),"gray")
    plt.show()
    print "you did this well: {0} (what units?)".format(sum(sum(cv2.absdiff(registered_img,img1)**2)))
    