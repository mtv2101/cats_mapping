### mattv@alleninstitute.org ###
### adapted from code by Nick Cain ###

import os
import shutil
import time
import numpy as np
import tables as tb
import pdb

def decimate_JCamF(input_file, output_file=None, 
                             dtype='<u2', 
                             header_length=116, 
                             length_digit=18, 
                             height_digit=82, 
                             width_digit=82,
                             verbose=False,
                             spatial_compression=2,
                             temporal_compression=1,
                             height = None,
                             width = None,
                             NFrames = None,
                             transpose = False):

    if output_file == None:
        output_file = input_file + '_' + str(spatial_compression) + '_' +  str(spatial_compression) + '_' + str(temporal_compression)+'.h5'
    
    # Get height, width, and length information from file header:
    f = open(input_file, 'rb')
    if height == None:
        f.seek(height_digit*np.dtype(dtype).itemsize, 0)
        height = int(np.fromfile(f, dtype=dtype, count=1)[0])
    if width == None:
        f.seek(width_digit*np.dtype(dtype).itemsize, 0)
        width = int(np.fromfile(f, dtype=dtype, count=1)[0])
    if NFrames == None:
        f.seek(length_digit*np.dtype(dtype).itemsize, 0)
        number_of_frames_from_header = np.fromfile(f, dtype=dtype, count=1)[0]
    else:
        number_of_frames_from_header = NFrames
    file_size = os.path.getsize(input_file)
    file_size_no_header = file_size - np.dtype(dtype).itemsize*header_length
    approx_number_of_frames = int(file_size_no_header*1./(np.dtype(dtype).itemsize*int(height)*int(width)))
    number_of_wrap_around = int(approx_number_of_frames//(2**(8*np.dtype(dtype).itemsize)))
    number_of_frames = number_of_wrap_around*(2**(8*np.dtype(dtype).itemsize)) + number_of_frames_from_header
    number_of_frame_chunks = int(number_of_frames*1./temporal_compression)

    print 'HEIGHT:',height
    print 'WIDTH:',width
    
    # Prepare a dictionary to dump to creash report, on error:
    debug_dict = {}
    debug_dict['height'] = str(height)
    debug_dict['width'] = str(width)
    debug_dict['number_of_frames_from_header'] = str(number_of_frames_from_header)
    debug_dict['file_size'] = str(file_size)
    debug_dict['file_size_no_header'] = str(file_size_no_header)
    debug_dict['approx_number_of_frames'] = str(approx_number_of_frames)
    debug_dict['number_of_wrap_around'] = str(number_of_wrap_around)
    debug_dict['number_of_frames'] = str(number_of_frames)
    debug_dict['spatial_compression'] = str(spatial_compression)
    debug_dict['temporal_compression'] = str(temporal_compression)

    # Double check:
    # try: assert height == width == 2048
    # except: raise Exception('Height and width not of expected size')    
    
    # Seek to beginning of data:
    f.seek(np.dtype(dtype).itemsize*header_length, 0)
    
    # Create output file as pytables EArray for compressible and extendible data storage
    # hdf5 file will extend along the time dimension so shape of dim0=0
    fd = tb.open_file(output_file, 'w')
    filters = tb.Filters(complevel=1, complib='blosc')
    if transpose:
        hdf5data = fd.create_carray(fd.root, 
					    	'data', 
					    	tb.UInt16Atom(), 
					    	filters=filters,
					    	shape=(int(height*1./spatial_compression)*int(width*1./spatial_compression), int(number_of_frames*1./temporal_compression)))
    else:            
        hdf5data = fd.create_earray(fd.root, 
					    	'data', 
					    	tb.UInt16Atom(), 
					    	filters=filters,
					    	shape=(0, int(height*1./spatial_compression),int(width*1./spatial_compression)))

    # Decimate by averaging:
    #t0 = time.time()
    max_val = 0
    try:
        # Loop across blocks of input data
        
        debug_dict['number_of_frame_chunks'] = str(number_of_frame_chunks)
        for frame_pair_ind in range(number_of_frame_chunks):
            
            # Print progress:
            if verbose == True:print "Progress: ", frame_pair_ind*1./number_of_frame_chunks
            #debug_dict['execution_time'] = seconds_to_str(time.time()-t0)
            debug_dict['frame_ind'] = str(2*frame_pair_ind)
            
            # Read in a block of data:
            data = np.fromfile(f,dtype=dtype,count=height*width*temporal_compression)
            data = data.reshape((temporal_compression,height,width)) + 1 # Corrects for single-bit errors in raw data, with line below
            
            # Accumulate data:
            D = np.zeros((int(height*1./spatial_compression), int(width*1./spatial_compression)), dtype=np.float64)
            for ii in range(spatial_compression):
                for jj in range(spatial_compression):
                    for kk in range(temporal_compression): 
                        D += data[kk, ii::spatial_compression,jj::spatial_compression]
        
            # Normalize:
            D -= spatial_compression*spatial_compression*temporal_compression     # Corrects for single-bit errors in raw data, with line above
            D /= spatial_compression*spatial_compression*temporal_compression
            
            # Double-check to make sure we won't have bit-depth problem:
            curr_max = D.max()
            if curr_max > 2**16:
                raise Exception('Single value too big')
            if curr_max > max_val:
                max_val = curr_max 
                debug_dict['max_val'] = str(max_val)
            
            # Round, and cast from float back to integer:
            D=np.round(D).astype(np.dtype(dtype))
            
            # Double Check top-left corner
            try:
                assert int(data[:,:spatial_compression,:spatial_compression].mean()), D[0,0]
            except:
                raise Exception('invalid value in frame pair %s, %s, %s' % (frame_pair_ind, int(data[:,:spatial_compression,:spatial_compression].mean()), D[0,0])) 
            
            # Write to file:            
            if transpose:
                D = D.reshape((np.prod(D.shape),))  
                D = np.expand_dims(D,axis=1)
                #pdb.set_trace()
                hdf5data[:,frame_pair_ind] = D.T
            else:
                D.reshape((np.prod(D.shape),))
                hdf5data.append(D[None])
        
    except Exception as e:  
#        debug_dict['error'] = str(e)
#        fdebug = open('debug_decimate_crash_report.json', 'w')
#        json.dump(debug_dict, fdebug, indent=2)
#        fdebug.close()
        print 'Error logged to debug_decimate_crash_report.json'
    
    f.close() 
    fd.close()
    
    
move = False
spatial_compression = 4
temporal_compression = 1
transpose = False

source = r"Y:\Matt\corrdata\150315\to_decimate"
JCamfiles = []
#Get a list of all large jcam files
# the size limitation will exclude single frame vascular images
# for f in os.listdir(source):
for root, dirs, files in os.walk(source):
    for f in files:
        if "JCam" in f and os.path.getsize(os.path.join(root,f)) > 20000000 and f[-3:] != '.h5':
            JCamfiles.append(os.path.join(root,f))
destination_folder = r'\\aibsdata2\nc-ophys\CorticalMapping\IntrinsicImageData'

print "Found the following files to decimate:"
for f in JCamfiles:
	print "  "+f
print ""

#decimate files listed in paths and delete the original files
t_deci = []
for path in JCamfiles:
    print 'decimating '+path+'...'
    t0 = time.time()
    decimate_JCamF(path,verbose=True,spatial_compression=spatial_compression,temporal_compression=temporal_compression, transpose=transpose)
    t_deci.append(time.time()-t0)
    t0 = time.time()
    print ""
    #deci(path,verbose=True,spatial_compression=16,temporal_compression=1)
    #t_deci.append(time.time()-t0)
    #os.remove(path)
    print 'end of decimation.'

if move == True:
    t_move = []
 #move all the folders in D:\ to destination_folder
    fds = next(os.walk('D:/'))[1]
    fds.remove('$RECYCLE.BIN')
    fds.remove('System Volume Information')
    
    print fds
    
    print ""
    for fd in fds:
        t0 = time.time()
        print 'Moving folder  D:/' + fd + '  to ',destination_folder
        shutil.copy2(os.path.join('D:/',fd),destination_folder)
        t_move.append(time.time()-t0)
    print 'Moving finished.'

print "decimation times:",t_deci
if move:
	print "move times:",t_move
print "=====DONE====="
print ""
raw_input("Press enter to exit...")