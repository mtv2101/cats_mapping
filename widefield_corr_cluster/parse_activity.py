#!/shared/utils.x86_64/python-2.7/bin/python

"""
Created on Thu Nov 2, 2015

@author: mattv
"""

import platform
import sys
if platform.system() == 'Linux': 
    sys.path.append(r'/data/nc-ophys/Matt/imagingbehavior')
import imaging_behavior as im
import numpy as np
from itertools import groupby
from operator import itemgetter
import matplotlib.pyplot as plt
from imaging_behavior.core.slicer import BinarySlicer
import pickle



def parse_activity(jsonpath, savepath, jcampath, win=5, rew_win=30, min_epoch = 30, JCam_srate = 100):

	"""  
	win. win*2 used to calculate mean runnign speed
	rew_win.  detect if a reward occurs within rew_win
	min_epoch(seconds). Minimum length of inactive or engaged epoch 
	"""

	session_jphys = im.load_session_from_manifest(jsonpath)	
	time_array_jphys = session_jphys.timeline.times['master_times']	

	srate = session_jphys.master_sampling_rate #10000
	sr_win = np.floor(win/2 * srate)
	mean_run = [] #store mean running speed per second here
	inactive = np.zeros(len(time_array_jphys)) #true or false for inactivity
	theta = session_jphys.timeline.values['running_speed_radians_per_sec']
	jphys_running = theta*session_jphys.running_radius

	# iterate per second of the recording
	for n in np.arange(0, time_array_jphys.shape[0], srate): #iterate every 1 second
	    mean_win = np.mean(jphys_running[(n-sr_win):(n+sr_win)]) #mean running speed during 10s window
	    if mean_win<1: #using 1 cm/s cutoff for inactivity per Niell and Stryker 2010
	        if n-sr_win < 0:
	            inactive[:n+sr_win] = 1
	        if n+sr_win > (len(inactive)-1):
	            inactive[(n-sr_win):] = 1
	        else:
	            inactive[(n-sr_win):(n+sr_win)] = 1
	    mean_run = np.append(mean_run, mean_win)

	inactive_per_second = np.nonzero(mean_run<1)[0]#Used as a check
	inactive = inactive.astype('i2')

	rew = session_jphys.reward_times
	timeline_sec = np.arange(0, time_array_jphys.shape[0], srate)
	    
	rew_epochs = np.zeros(timeline_sec.shape, dtype=bool)
	active=np.zeros(len(time_array_jphys))
	for n,t in enumerate(timeline_sec): 
	    val = [True for r in rew if r-rew_win < n < r+rew_win] # True if a reward occurs within rew_win of that second
	    if val:
	        rew_epochs[n] = True
	        if t-srate/2 > 0:   
	            active[(t-srate/2):(t+srate/2)] = np.ones(srate)
	active = active.astype('i2')

	# check for active times (i.e rewards) within inactive epoch and remove them
	overlap = np.equal(inactive, active)
	inactive[overlap]=0

	srate_min_epoch = srate * min_epoch
	long_inactive = []
	epochs=[]
	for k, g in groupby(inactive):
	    epochs.append(list(g))
	for row in epochs:
	    if sum(row)>=srate_min_epoch:
	        long_inactive.append(row)
	    else:
	        long_inactive.append(np.zeros(len(row)))
	inactive_epochs = map(sum,long_inactive)
	valid_inactive = np.hstack(long_inactive)
	if valid_inactive == []:
	    print 'no periods of inactivity found, try changing parameters'

	# format inactive to int frames
	Jcam_inactive = np.nonzero(valid_inactive)[0]
	downsample_slicer = np.arange(0, Jcam_inactive.shape[0], int(srate)/JCam_srate)
	Jcam_inactive = np.floor(Jcam_inactive[downsample_slicer]/(int(srate)/JCam_srate))
	Jcam_inactive = Jcam_inactive.astype('int32')

	Jcam_active = np.nonzero(active)
	downsample_slicer = np.arange(0, Jcam_active[0].shape[0], int(srate)/JCam_srate)
	Jcam_active = np.floor(Jcam_active[0][downsample_slicer]/(int(srate)/JCam_srate))
	Jcam_active = Jcam_active.astype('int32')

	# get number of JCam frames, truncate JPhys detected frames to match
	frames = BinarySlicer(str(jcampath))
	nframes = np.shape(frames)[0]
	Jcam_active = Jcam_active[Jcam_active<nframes]
	Jcam_inactive = Jcam_inactive[Jcam_inactive<nframes]

	print 'there were ' + str(len(Jcam_active)/JCam_srate) + ' seconds of activity'
	print 'there were ' + str(len(Jcam_inactive)/JCam_srate) + ' seconds of inactivity in recording with the longest lasting ' + str(max(inactive_epochs)/srate) + ' seconds'

	pickle.dump([Jcam_inactive, Jcam_active], open(savepath + '.pkl', 'w'))

	return Jcam_inactive, Jcam_active