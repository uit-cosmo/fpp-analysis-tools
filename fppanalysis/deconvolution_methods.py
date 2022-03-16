# Methods for performing Richardson-Lucy deconvolution.
# A kernel (pulse shape) is needed.
#
# First, use RL_gauss_deconvolve to perform the deconvolution.
# More iterations are better.
# Check that the result falls to zero during quiet periods, or amplitudes
# may not be correctly calculated.
#
# Then, use find_amp_ta to calculate the peaks. The default values work OK.
# Noise is handeled by increasing window_length or order in find_amp_ta.


def RL_gauss_deconvolve(sig, kern, iterlist, init=None, cutoff=1e-10, sf=1):
    """
    Use: RL_gauss_deconvolve(sig,kern, iterlist, init=None, cutoff=1e-10)
    Performs the Richardson-Lucy deconvolution for normally distributed noise.
    See https://en.wikipedia.org/wiki/Richardson%E2%80%93Lucy_deconvolution
    and https://arxiv.org/abs/1802.05052.

    Input:
        sig: signal to be deconvolved ............................. 1D np array
        kern: deconvolution kernel ................................ 1D np array
        iterlist: the number of iterations. ....................... int or
              If this is a list, the deconvolution result           list of int
              is returned for each element in iterlist, see below.
        init: initial array guess. Leave blank for all zeros. ..... 1D np array
        cutoff: for avoiding divide by zero errors. ............... float
        sf: scale factor which is multiplied with b condition...... float > 0, default = 1

    Output:
        res: result array. NxM, where N=len(sig) and M=len(iterlist)   np array
        err: mean absolute difference between iterations .......... 1D np array
    
    WARNING:
    For estimating the pulse shape, you need to ensure you have an odd number of data points when generating synthetic data.
    Do a check like the following before putting S into the sig argument.
    if (len(S) % 2) == 0:
        S = S[:-1]
        T = T[:-1]
    """
    import numpy as np
    from tqdm import tqdm
    from scipy.signal import fftconvolve

    if init is None:
        update0 = np.ones(sig.size)
        update1 = np.ones(sig.size)
    else:
        update0 = np.copy(init)
        update1 = np.copy(init)

    sigtmp = np.copy(sig)
    kerntmp = np.copy(kern)

    if type(iterlist) is int:
        iterlist = [
            iterlist,
        ]

    err = np.zeros(iterlist[-1] + 1)
    err[0] = np.sum((sigtmp - fftconvolve(update0, kerntmp, "same")) ** 2)
    res = np.zeros([sig.size, len(iterlist)])

    kern_inv = kerntmp[::-1]
    sigconv = fftconvolve(sigtmp, kern_inv, "same")
    kernconv = fftconvolve(kerntmp, kern_inv, "same")

    # To ensure we have non-negative iterations we apply a condition which is dependent on sigconv
    # If signconv is negative then b = cutoff - sf*np.amin(sigconv). Cutoff avoids division by 0.
    # If sigconv is positive, let b = cutoff to ensure non-zero division.
    if np.amin(sigconv) < 0:
        b_min = np.amin(sigconv)
        b = cutoff - sf * b_min
    else:
        b = cutoff

    index_array = np.arange(sigtmp.size)
    count = 0

    for i in tqdm(range(1, iterlist[-1] + 1), position=0, leave=True):
        # If an element in the previous iteration is very close to zero,
        # the same element in the next iteration should be as well.
        # This is handeled numerically by setting all elements <= cutoff to 0
        # and only performing the interation on those elements > cutoff.

        tmp = fftconvolve(update0, kernconv, "same")

        update1 = (update0 * (sigconv + b)) / (tmp + b)

        err[i] = np.sum((sigtmp - fftconvolve(update1, kerntmp, "same")) ** 2)
        update0[:] = update1[:]

        if i == iterlist[count]:
            print("i = {}".format(iterlist[count]), flush=True)
            res[:, count] = update1[:]
            count += 1

    return res, err


def find_amp_ta_3nn(D, tb=None, **kwargs):
    """
    Find amplitudes and arrival times of the deconvoved signal D
    using scipy.signal.find_peaks.
    
    Use: ta,amp = find_amp_ta(D, tb=None, **kwargs)
    Estimates arrival times and amplitudes
    of the FPP from the deconvolved signal D.
    Input:
        D: result of deconvolution ............... numpy array
        tb: (optional) time array ................ numpy array
        **kwargs ................................. passed to find_peaks
    Output:
        ta: estimated location of arrivals ....... numpy array
        amp: estimated amplitudes ................ numpy array
        
    If tb is given, ta are the arrival times in tb. 
    If tb is not given, ta are the peak locations in D.
    
    By default, this is a pure 3-point maxima. 
    In the presence of noise, we suggest using one of the following
    keywords in order to select for the peaks:
        height: required height of peak
        prominence: required prominence of peak
            may require setting wlen as well, for large arrays.
    
    In order to take the entire mass of each peak of D into account,
    the amplitudes are estimated by summing from one minima between two peaks
    to the minima between the next two peaks. The value of the minima is 
    divided proportionally between the two peaks, according to their height.
    ---min---peak----min----peak----min---
    ---][--sum range-][--sum range--][----
    """
    import numpy as np
    from scipy.signal import find_peaks

    # Find_peaks discounts the endpoints
    Dtmp = np.zeros(D.size + 2)
    Dtmp[1:-1] = D[:]

    peak_loc = find_peaks(Dtmp, **kwargs)[0]

    amp = Dtmp[peak_loc]

    if tb is None:
        return peak_loc - 1, amp
    else:
        return tb[peak_loc - 1], amp
