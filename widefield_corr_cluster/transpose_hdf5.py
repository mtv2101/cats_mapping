#!/shared/utils.x86_64/python-2.7/bin/python

import numpy as np
import tables as tb
import timeit
import os
import pdb
from skimage.measure import block_reduce
    
def transpose(filepath, output_file, start=0, stop=1000000, chunk=8, method=1, dsfactor=4):  
    
    ''' 
    this code will open a [t, y*x] hdf5 movie and transpose it so 
    that the output dims are [y*x, t].  Transposition is applied to chunks
    of rows of the input image, and the output is saved as an extensible 
    ptyables array
    '''

    # load data
    open_tb = tb.open_file(filepath, 'r')
    raw_mov = open_tb.root.data

    mov = downsample(raw_mov, dsfactor)
    
    print 'number of frames = ' + str(mov.shape[0]) + '\n'
    if stop > mov.shape[0]:
        frames = range(start, mov.shape[0])
    else:
        frames = range(start, stop)
    print str(len(frames)) + ' frames will be detrended\n'
    
    tdim = len(frames)
    sdim = mov.shape[1]*mov.shape[2]
    
    
    if method==1:
        # open output file which will be a space x time transposition of the movie
        fd = tb.openFile(output_file, 'w')
        filters = tb.Filters(complevel=1, complib='blosc')
        mov_transpose = fd.createEArray(fd.root, 
                        'data', 
                        tb.UInt16Atom(), 
                        filters=filters,
                        expectedrows=sdim,
                        shape=(0, int(tdim)))

        # transpose each row individually
        print str(int(np.ceil(mov.shape[1]/chunk))) + ' chunks will be transposed\n'
        start_time = timeit.default_timer()
        for nn,n in enumerate(np.arange(0, mov.shape[1], chunk)):
            print 'chunk # ' + str(nn)
            # transpose the array
            transposed_mat = mov[frames,n:n+chunk,:].reshape([len(frames), chunk*mov.shape[2]]).T
    
            mov_transpose.append(transposed_mat)
        
        transpose_time = timeit.default_timer() - start_time
        print 'transposition took ' + str(transpose_time) + ' seconds\n'
        
    if method==2:
        # transpose each frame individually
        fd = tb.openFile(output_file, 'w')
        mov_transpose = fd.createCArray(fd.root, 
                        'data', 
                        tb.UInt16Atom(), 
                        shape=(int(sdim), int(tdim)))
                        
        start_time = timeit.default_timer()
        for nn,n in enumerate(frames):
            #print 'frame # ' + str(nn)
            transposed_row = mov[n,:,:].reshape([mov.shape[1]*mov.shape[2]]).T
            mov_transpose[:,nn] = transposed_row
        transpose_time = timeit.default_timer() - start_time
        print 'transposition took ' + str(transpose_time) + ' seconds\n'
        
    if method==3:
        # pull all rows but accumulate chunksize number of columns from those rows. 
        # Transpose these columns as resave as output array
        fd = tb.openFile(output_file, 'w')
        filters = tb.Filters(complevel=1, complib='blosc')
        mov_transpose = fd.createEArray(fd.root, 
                        'data', 
                        tb.UInt16Atom(), 
                        filters=filters,
                        expectedrows=sdim,
                        shape=(0, int(tdim)))
                        
        # transpose each row individually
        print str(int(np.ceil(mov.shape[1]/chunk))) + ' chunks will be transposed'
        start_time = timeit.default_timer()
        #pdb.set_trace()
        row_accum = np.zeros([mov.shape[0], chunk])
        for kk,k in enumerate(np.arange(0, mov.shape[1], chunk)):
            for nn,n in enumerate(frames):
                #print 'chunk # ' + str(nn)
                # transpose the array
                row = mov[n,:,:].reshape([mov.shape[1]*mov.shape[2]])
                row_accum[n,:] = row[k:k+chunk]
    
            mov_transpose.append(row_accum.T)
        
        transpose_time = timeit.default_timer() - start_time
        print 'transposition took ' + str(transpose_time) + ' seconds\n'
        
    fd.close()
    open_tb.close()


def downsample(mov, dsfactor):
    mov_ds = np.zeros([mov.shape[0], mov.shape[1]/dsfactor, mov.shape[2]/dsfactor])
    for i in range(mov.shape[0]):
        mov_ds[i,:,:] = block_reduce(mov[i], block_size=(dsfactor,dsfactor), func=np.mean)
    return mov_ds
   

if __name__ == "__main__": 
    
    transpose(r'F:\150729-M187476\010101JCamF102_16_16_1_64x64.h5', r'F:\150729-M187476\transpose_test3.h5', start=0, stop=1000, chunk=8, method=3)