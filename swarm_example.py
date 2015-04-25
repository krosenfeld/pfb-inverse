import numpy as np
import scipy.linalg as la
import matplotlib.pyplot as plt
from time import time
import resource
from datetime import datetime
import cross_corr

#from cvxopt import matrix
#import pfb
#from cvxopt.lapack import pbsv

# Script based off of 
# http://nbviewer.ipython.org/github/jrs65/pfb-inverse/blob/master/notes.ipynb

# Changes:
#   in inervse_pfb:
#     la.solveh_banded produced an error:
#     ** On entry to DPBSV parameter number  8 had an illegal value
#     This is a problem for the scipy 0.7.2, but should be fixed for later
#     versions.  I have opted to use pbsv from cvxopt instead.
#

def sinc_window(ntap, lblock):
    """Sinc window function.
    
    Parameters
    ----------
    ntaps : integer
        Number of taps.
    lblock: integer
        Length of block.
        
    Returns
    -------
    window : np.ndarray[ntaps * lblock]
    """
    coeff_length = np.pi * ntap
    coeff_num_samples = ntap * lblock
    
    # Sampling locations of sinc function
    X = np.arange(-coeff_length / 2.0, coeff_length / 2.0,
                  coeff_length / coeff_num_samples)
    
    # np.sinc function is sin(pi*x)/pi*x, not sin(x)/x, so use X/pi
    return np.sinc(X / np.pi)


def sinc_hanning(ntap, lblock):
    """Hanning-sinc window function.
    
    Parameters
    ----------
    ntaps : integer
        Number of taps.
    lblock: integer
        Length of block.
        
    Returns
    -------
    window : np.ndarray[ntaps * lblock]
    """
    
    return sinc_window(ntap, lblock) * np.hanning(ntap * lblock)

def hamming(ntap, lblock):
    """Hamming window function.
    
    Parameters
    ----------
    ntaps : integer
        Number of taps.
    lblock: integer
        Length of block.
        
    Returns
    -------
    window : np.ndarray[ntaps * lblock]
    """
    
    return np.hamming(ntap * lblock)

def pfb(timestream, nfreq, ntap=4, window=hamming):
    """Perform the SWARM PFB on a timestream.
    
    Parameters
    ----------
    timestream : np.ndarray
        Timestream to process
    nfreq : int
        Number of frequencies we want out (probably should be odd
        number because of Nyquist)
    ntaps : int
        Number of taps.

    Returns
    -------
    pfb : np.ndarray[:, nfreq]
        Array of PFB frequencies.
    """
    
    # Number of samples in a sub block
    lblock = 2 * (nfreq - 1)
    
    # Number of blocks
    nblock = timestream.size / lblock - (ntap - 1)
    
    # Initialise array for spectrum
    spec = np.zeros((nblock, nfreq), dtype=np.complex128)
    
    # Window function
    w = window(ntap, lblock)
    
    # Iterate over blocks and perform the PFB
    for bi in range(nblock):
        # Cut out the correct timestream section
        ts_sec = timestream[(bi*lblock):((bi+ntap)*lblock)].copy()
        
        # Perform a real FFT (with applied window function)
        ft = np.fft.rfft(ts_sec * w)
        
        # Choose every n-th frequency
        spec[bi] = ft[::ntap]
        
    return spec

def pfb_timestream_fullmatrix(ntime, nfreq, ntap=4, window=hamming):
    
    # Number of samples in a sub-block
    lblock = 2*(nfreq - 1)
    
    # Number of blocks in timestream
    nblocks = ntime / lblock
    
    # Number of blocks in PFB
    npfb = nblocks - ntap + 1
    
    # Initialise matrix
    mat = np.zeros((npfb, lblock, nblocks, lblock))
    
    # Window function
    w = window(ntap, lblock)
    
    # Iterate over PFB blocks setting the elements
    for bi in range(npfb):
        for si in range(lblock):
            for ai in range(ntap):
                mat[bi, si, bi+ai, si] = w[si + ai * lblock]
    
    return mat

# Routine wrapping Lapack dgbmv
def band_mv(A, kl, ku, n, m, x, trans=False):
    """
    Performs the matrix-vector operation

    y := alpha*A**T*x 

    where alpha and beta are scalars, x is a vector and A is an
    m by n band matrix, with kl sub-diagonals and ku super-diagonals.
    """
    import dgbmv
    
    y = np.zeros(n if trans else m, dtype=np.float64)
   
    lda = kl + ku + 1
    
    #dgbmv.dgbmv('T' if trans else 'N', m, kl, ku, 1.0, A, x, 1, 0.0, y, 1)
    dgbmv.dgbmv('T' if trans else 'N', m, n, kl, ku, 1.0, A, x, 1, 0.0, y, 1)
    
    return y

def inverse_pfb(ts_pfb, ntap, window=hamming):
    """Invert the SWARM PFB timestream.
    
    Parameters
    ----------
    ts_pfb : np.ndarray[nsamp, nfreq]
        The PFB timestream.
    ntap : integer
        The number of number of blocks combined into the final timestream.
    window : function (ntap, lblock) -> np.ndarray[lblock * ntap]
        The window function to apply to each block.
    """
    
    # Inverse fourier transform to get the pseudo-timestream
    pseudo_ts = np.fft.irfft(ts_pfb, axis=-1)
    
    # Transpose timestream
    pseudo_ts = pseudo_ts.T.copy()
    
    # Pull out the number of blocks and their length
    lblock, nblock = pseudo_ts.shape
    ntsblock = nblock + ntap - 1
    
    # Coefficients for the P matrix
    coeff_P = window(ntap, lblock).reshape(ntap, lblock)  # Create the window array

    # Coefficients for the PP^T matrix
    coeff_PPT = np.array([ (  coeff_P[:, np.newaxis, :]
                            * coeff_P[np.newaxis, :, :] ).diagonal(offset=k).sum(axis=-1)
                           for k in range(ntap) ])
    
    rec_ts = np.zeros((lblock, ntsblock), dtype=np.float64)
    
    for i_off in range(lblock):

        # Create band matrix representation of P
        band_P = np.zeros((ntap, ntsblock), dtype=np.float64)
        band_P[:] = coeff_P[::-1, i_off, np.newaxis]

        # Create band matrix representation of PP^T (symmetric)
        band_PPT = np.zeros((ntap, nblock), dtype=np.float64)
        band_PPT[:] = coeff_PPT[::-1, i_off, np.newaxis]

        # Solve for intermediate vector
        yh = la.solveh_banded(band_PPT, pseudo_ts[i_off])

        # Project into timestream estimate
        rec_ts[i_off] = band_mv(band_P, 0, 3, ntsblock, nblock, yh, trans=True)

	# using cvxopt
        # Solve for intermediate vector
        #A = matrix(band_PPT)
        #B = matrix(pseudo_ts[i_off])
        #pbsv(A,B,uplo='U')

        # Project into timestream estimate
        #rec_ts[i_off] = band_mv(band_P, 0, 3, ntsblock, nblock, B, trans=True)
        
    # Transpose timestream back
    rec_ts = rec_ts.T.copy()
    
    return rec_ts

############################################################################
############################################################################

# VDIF frame size
FRAME_SIZE_BYTES = 1056

# SWARM related constants, should probably be imported from some python
# source in the SWARM git repo
SWARM_XENG_PARALLEL_CHAN = 8
SWARM_N_INPUTS = 2
SWARM_N_FIDS = 8
SWARM_TRANSPOSE_SIZE = 128
SWARM_CHANNELS = 2**14
SWARM_CHANNELS_PER_PKT = 8
SWARM_PKTS_PER_BCOUNT = SWARM_CHANNELS/SWARM_CHANNELS_PER_PKT
SWARM_SAMPLES_PER_WINDOW = 2*SWARM_CHANNELS
SWARM_RATE = 2496e6

if __name__ == "__main__":

  plt.close('all')

  # setup 
  #window = hamming
  window = sinc_hanning
  num_taps = 4
  num_bframes = 1
  num_samples = SWARM_SAMPLES_PER_WINDOW * (SWARM_TRANSPOSE_SIZE*num_bframes + num_taps - 1)
  #num_snapshots = 1
  #num_samples = SWARM_SAMPLES_PER_WINDOW * (num_snapshots + num_taps - 1)
  num_freq = SWARM_SAMPLES_PER_WINDOW/2 + 1 # include nyquist for now

  # Generate a white noise timestream
  tic = datetime.now()
  ts = np.random.standard_normal(num_samples)
  print 'generating time series takes:', (datetime.now()-tic).total_seconds()

  # Perform the PFB
  tic = datetime.now()
  spec_pfb = pfb(ts,num_freq,ntap=num_taps,window=window)
  print 'performing PFB takes:', (datetime.now()-tic).total_seconds()


  # Perform the inverse
  tic = datetime.now()
  rts = inverse_pfb(spec_pfb, 4, window=window)
  print 'inverse pfb takes:', (datetime.now()-tic).total_seconds()

  # show a small section of the PFB (real part)
  plt.figure()
  plt.imshow(spec_pfb[:10].real, interpolation='nearest', aspect='auto', cmap='RdBu')
  plt.colorbar()  

  # show time series residuals
  plt.figure()
  plt.plot((ts - rts.ravel())[:np.min([ 5000000,ts.size ])])
  plt.ylabel('residual')

  # plot standard deviation of residuals for blocks of data (128)
  block_res = ts.reshape(-1, rts.shape[-1]) - rts
  plt.figure()
  plt.title("Standard deviation of residuals")
  plt.semilogy(block_res.std(axis=1))
  plt.xlabel("Snapshots")

  # cross correlate 
  s_0x1,S_0x1,s_peaks = cross_corr.corr_FXt(ts,rts.ravel())
  plt.figure()
  plt.stem(s_0x1[:20])
  plt.ylabel('Corr coeff')
  plt.xlabel("lag")
  plt.xlim([-0.4,20])

  # show figures
  plt.ion()
  plt.show()

  # print stats:
  print 'Memory usage: %d bytes (%8.6f GB).' % (resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/2.0**20)
  print 'mean squared error: {0:g}'.format(np.mean((ts - rts.ravel())**2))
  print 'using {0:g} ms = {1:g} GB'.format(num_samples / SWARM_RATE * 1e3, ts.nbytes / 1e9)
  print 'corr coef: {0:g}'.format(s_0x1[0])


