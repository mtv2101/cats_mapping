#!/shared/utils.x86_64/python-2.7/bin/python

"""
3/3/2017
mattv@alleninstitute.org
"""


import os
import timeit
import numpy as np
import tables as tb
from skimage.measure import block_reduce
from scipy.signal import resample, butter, filtfilt
from scipy.interpolate import interp1d
from scipy.stats import linregress 



def import_movies(movpath1, movpath2, numframes=False):

	print 'opening ' + str(movpath1)
	open_mov1 = tb.open_file(movpath1, 'r')
	mov1 = open_mov1.root.data
	print str(mov1.shape) + ' ' + str(mov1.dtype)

	print 'opening ' + str(movpath2)
	open_mov2 = tb.open_file(movpath2, 'r')
	mov2 = open_mov2.root.data
	print str(mov2.shape) + ' ' + str(mov2.dtype)

	if numframes:
		mov1 = mov1[:numframes, :, :]
		mov2 = mov2[:numframes/2, :, :]

	return mov1, mov2


def import_jphys(jphyspath):
	# assume mov2 contains two channels imaged 25Hz each 

	JPhysFile = np.fromfile(jphyspath, dtype=np.dtype('>f4'), count=-1)
	channelNum = 3
	channelLength = len(JPhysFile) / channelNum
	JPhysFile = JPhysFile.reshape([channelLength, channelNum])
	print 'openend JPhys file ' + str(jphyspath)

	# detrend JPhys traces (there are sometimes DC drift in these traces)
	for n in range(channelNum):
		xtrace = np.arange(0,JPhysFile[:, n].shape[0], 1)
		slope, intercept, r_value, p_value, std_err = linregress(xtrace,JPhysFile[:, n])
		JPhysFile[:,n] -= ((slope*xtrace)+intercept)

	return JPhysFile


def downsample_movie(mov, desamp):
	start_time = timeit.default_timer()

	movds = np.zeros([mov.shape[0], mov.shape[1]/desamp, mov.shape[2]/desamp]).astype('u2')
	for i in range(mov.shape[0]):
		movmean = block_reduce(mov[i,:,:], block_size=(desamp,desamp), func=np.mean)
		movds[i,:,:] = movmean.astype('u2')

	run_time = timeit.default_timer() - start_time
	print 'spatial downsampling took ' + str(run_time) + ' seconds'

	return movds


def vsync_timing(jphys, thresh1, thresh2):
	# crop movies from two cameras to approximately match.  Measure offset of start and stop times between the cameras.

	vsync1 = jphys[:,2]
	vsync_high1 = [t for t,v in enumerate(vsync1) if v < thresh1 and vsync1[t-1] > thresh1]
	vsync_high1_filt = [stim for s,stim in enumerate(vsync_high1[:-1]) if (vsync_high1[s+1] - vsync_high1[s]) > 50]
	print str(len(vsync_high1_filt)) + ' vsyncs detected from fluorescence camera'

	vsync2 = jphys[:,1]
	vsync_high2 = [t for t,v in enumerate(vsync2) if v < thresh2 and vsync2[t-1] > thresh2]
	vsync_high2_filt = [stim for s,stim in enumerate(vsync_high2[:-1]) if (vsync_high2[s+1] - vsync_high2[s]) > 50]
	print str(len(vsync_high2_filt)) + ' vsyncs detected from reflectance camera'

	# last four vsync pulses are extraneous
	vsync1_start = vsync_high1_filt[4]
	vsync1_stop = vsync_high1_filt[-4]
	# reflectance camera has 2 initialization pulses because it is not recieving a trigger-start signal 
	vsync2_start = vsync_high2_filt[0]
	vsync2_stop = vsync_high2_filt[-4]

	common_start = np.max([vsync1_start, vsync2_start])
	common_stop = np.min([vsync1_stop, vsync2_stop])

	cam1_times = [frame for f,frame in enumerate(vsync_high1_filt) if frame > common_start and frame < common_stop]
	cam1_frames = [f for f,frame in enumerate(vsync_high1_filt) if frame > common_start and frame < common_stop]
	# ensure overhang of cam2 around start and stop of cam1
	cam2_times = [frame for f,frame in enumerate(vsync_high2_filt) if frame > common_start-4000. and frame < common_stop+2000.]

	print 'last fluorescence frame with cooresponding reflectance frame is frame ' + str(cam1_frames[-1])

	return cam1_times, cam2_times, cam1_frames

def deinterleave(mov):
	# seperate alternating frames in a movie
	mov1 = mov[0::2]
	mov2 = mov[1::2]
	return mov1, mov2

def butter_lowpass(sig, fs, corner, order=3):
    nyq = 0.5 * fs
    c = corner / nyq
    b,a = butter(order, c, btype='lowpass', analog=False)
    y1 = filtfilt(b, a, sig, axis=0)
    return y1

def resample(t,y,new_t):
    f = interp1d(t,y,axis=0,bounds_error=False,fill_value='extrapolate')
    return f(new_t)

def interpolate_movies(cam1, cam2, cam1_times, cam2_times, cam1_frames, filt):
	# interpolate cam2 to cam1's timebase
	# if start of cam2 is delayed compared to cam1 ensure cam2 frames overlap the start and stop of cam1 to reduce errors interpolating cam2 to cam1

	cam2_dt = 25.0

	start_time = timeit.default_timer()
	print 'aligning ' + str(len(cam1_frames)) + ' out of ' + str(cam1.shape[0]) + ' fluorescence frames'

	print cam1_frames[-1]
	if cam1.ndim > 1:
		cam1_cropped = cam1[cam1_frames,:,:]
	else:
		cam1_cropped = cam1[cam1_frames]

	cam2_chan1, cam2_chan2 = deinterleave(cam2)

	cam2_chan1_frames, cam2_chan2_frames = deinterleave(cam2_times)

	print 'Reflectance chan1 is ' + str(len(cam2_chan1_frames)) + ' frames'
	print 'Reflectance chan2 is ' + str(len(cam2_chan2_frames)) + ' frames'

	# 5Hz lowpass butter
	if filt:
		cam2_chan1_filt = butter_lowpass(cam2_chan1, cam2_dt, cam2_dt/5., order=5)
		cam2_chan2_filt = butter_lowpass(cam2_chan2, cam2_dt, cam2_dt/5., order=5)
	else:
		cam2_chan1_filt = cam2_chan1
		cam2_chan2_filt = cam2_chan2
	del cam2_chan1, cam2_chan2

	run_time = timeit.default_timer() - start_time
	print 'deinterleaving and filtering took ' + str(run_time) + ' seconds'
	start_time = timeit.default_timer()

	cam1_duration = cam1_times[-1] - cam1_times[0]
	#print 'cam1 is ' + str(len(cam1_times)) + ' samples and ' + str(cam1_duration/10000.) + ' seconds long'
	cam1_srate  = len(cam1_times) / (cam1_duration/10000.) # samples/sec
	print 'Fluorescence sampled at ' + str(cam1_srate) + ' Hz'

	chan1_duration = cam2_chan1_frames[-1] - cam2_chan1_frames[0]
	chan1_frames = int(np.ceil(chan1_duration/cam1_srate)) # convert to cam1 sample rate
	chan2_duration = cam2_chan2_frames[-1] - cam2_chan2_frames[0]
	chan2_frames = int(np.ceil(chan2_duration/cam1_srate)) # convert to cam1 sample rate

	# interpolate to cam1 timebase
	new_frames = len(cam1_times)
	resampt = np.arange(0, new_frames/cam1_srate, 1.0/cam1_srate)
	ydim = cam1_cropped.shape[1]
	xdim = cam1_cropped.shape[2]
	t1 = cam2_chan1_filt.shape[0]
	t2 = cam2_chan2_filt.shape[0]

	print 'interpolating reflectance chan 1 from ' + str(t1) + ' frames to ' + str(new_frames) + ' frames'
	cam2_chan1_filt = cam2_chan1_filt.reshape([t1, ydim*xdim])
	ts1 = np.arange(0, t1/cam2_dt, 1.0/cam2_dt)	
	interp_cam2_chan1 = [resample(ts1, cam2_chan1_filt[:,n], resampt) for n in range(ydim*xdim)]
	interp_cam2_chan1 = np.array(interp_cam2_chan1).T
	interp_cam2_chan1 = interp_cam2_chan1.reshape([interp_cam2_chan1.shape[0], ydim, xdim])

	print 'interpolating reflectance chan 2 from ' + str(t2) + ' frames to ' + str(new_frames) + ' frames'
	cam2_chan2_filt = cam2_chan2_filt.reshape([t2, ydim*xdim])
	ts2 = np.arange(0, t2/cam2_dt, 1.0/cam2_dt)	
	interp_cam2_chan2 = [resample(ts2, cam2_chan2_filt[:,n], resampt) for n in range(ydim*xdim)]
	interp_cam2_chan2 = np.array(interp_cam2_chan2).T
	interp_cam2_chan2 = interp_cam2_chan2.reshape([interp_cam2_chan2.shape[0], ydim, xdim])

	run_time = timeit.default_timer() - start_time
	print 'temporal interpolation took ' + str(run_time) + ' seconds'

	# find order of interleaving and calculate offset to nearest common cam1 frame
	if cam2_chan1_frames[0] > cam2_chan2_frames[0]:
		interp_cam2_chan2 = interp_cam2_chan2[2:]
		print 'indexed cam2chan1 ahead by ' + str(20) + ' ms to match cam2chan2'
		offset_error = (cam1_times[0] - cam2_chan1_frames[0])/100.

	elif cam2_chan2_frames[0] > cam2_chan1_frames[0]:
		interp_cam2_chan1 = interp_cam2_chan1[2:]
		print 'indexed cam2chan2 ahead by ' + str(20) + ' ms to match cam2chan1'
		offset_error = (cam1_times[0] - cam2_chan2_frames[0])/100.

	print 'reflectance channels ' + str(offset_error*10.) + ' ms behind fluorescence channel'
	# cut off starting frames to match start time of fluorescence trace  
	global_shift = int(np.around(offset_error))
	interp_cam2_chan1 = interp_cam2_chan1[global_shift:]
	interp_cam2_chan2 = interp_cam2_chan2[global_shift:]
	print 'shifted reflectance channels by ' + str(offset_error*10.) + ' ms or ' + str(global_shift) + ' JCam frames'

	# shorten movies to match length of shortest member
	print 'fluorescence = ' + str(cam1_cropped.shape[0])
	print 'reflectance chan1 = ' + str(interp_cam2_chan1.shape[0])
	print 'reflectance chan2 = ' + str(interp_cam2_chan2.shape[0])
	movie_len = np.min([cam1_cropped.shape[0], interp_cam2_chan1.shape[0], interp_cam2_chan2.shape[0]])
	cam1_cropped = cam1_cropped[:movie_len]
	interp_cam2_chan1 = interp_cam2_chan1[:movie_len]
	interp_cam2_chan2 = interp_cam2_chan2[:movie_len]

	return cam1_cropped, interp_cam2_chan1, interp_cam2_chan2

def align_2cam(movpath1, movpath2, jphyspath, ycord=None, xcord=None, desamp=1, numframes=False, filt=False, thresh1=1.0, thresh2=1.0):
	### mov1 is fluorescence, mov2 is reflectance
	## THIS IS THE OPPOSITE OF CONVENTION IN HEMO_2CAM.PY!!!

	mov1, mov2 = import_movies(movpath1, movpath2, numframes=numframes)

	jphys = import_jphys(jphyspath)

	mov1_times, mov2_times, mov1_frames = vsync_timing(jphys, thresh1, thresh2)

	if desamp != 1:
		mov1_ds = downsample_movie(mov1, desamp=desamp)
		mov2_ds = downsample_movie(mov2, desamp=desamp)
	else:
		mov1_ds = mov1
		mov2_ds = mov2
	del mov1, mov2

	if ycord and xcord:
		cam1, cam2_chan1, cam2_chan2 = interpolate_movies(mov1_ds[:,ycord,xcord], mov2_ds[:,ycord,xcord], mov1_times, mov2_times, mov1_frames, filt)
	else:
		cam1, cam2_chan1, cam2_chan2 = interpolate_movies(mov1_ds, mov2_ds, mov1_times, mov2_times, mov1_frames, filt)

	return cam1, cam2_chan1, cam2_chan2