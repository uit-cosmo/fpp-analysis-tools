# Methods for performing Richardson-Lucy deconvolution.
# A kernel (pulse shape) is needed.
#
# First, use RL_gauss_deconvolve to perform the deconvolution.
# More iterations are better.
# Check that the result falls to zero during quiet periods, or amplitudes
# may not be correctly calculated.
#
# Then, use three_point_maxima function to find the peaks.
# Noise is handeled by setting a height threshold related to
# the intermittency parameter (gamma), noise to signal ratio (epsilon) and the mean amplitude of the signal (<A>)
# where this is <A>*square_root(gamma*epsilon)


def RL_gauss_deconvolve(
    signal,
    kern,
    iteration_list,
    initial_guess=None,
    cutoff=1e-10,
    scale_factor=1,
    gpu=False,
):
    """
    Use: Performs the Richardson-Lucy (RL) deconvolution for normally distributed noise.
    See https://en.wikipedia.org/wiki/Richardson%E2%80%93Lucy_deconvolution
    and https://arxiv.org/abs/1802.05052.
    Input:
        signal: signal to be deconvolved ...................................... 1D np array
        kern: deconvolution kernel, this can be the pulse shape or the forcing  1D np array
        iteration_list: the number of iterations. ............................. int or list of int
              If this is a list, the deconvolution result 
              is returned for each element in iteration_list, see below.
        initial_guess: initial array guess. Leave blank for all zeros. ........ 1D np array
        cutoff: for avoiding divide by zero errors. ........................... float
        scale_factor: scale factor which is multiplied with b condition........ float > 0, default = 1
        gpu: use GPU-accelerated version....................................... bool
    Output:
        result: result array. NxM, where N=len(signal) and M=len(iteration_list)   np array
        error: mean absolute difference between iterations .......... 1D np array
    
    WARNING:
    For estimating the pulse shape, you need to ensure you have an odd number of data points when generating synthetic data.
    Do a check like the following before putting S into the signal argument.
    if (len(S) % 2) == 0:
        S = S[:-1]
        T = T[:-1]
    """
    from tqdm import tqdm

    if gpu:
        import cupy as xp
        from cusignal.convolution.convolve import fftconvolve
    else:
        import numpy as xp
        from scipy.signal import fftconvolve

    if gpu:
        pool = cp.cuda.MemoryPool(cp.cuda.malloc_managed)
        cp.cuda.set_allocator(pool.malloc)

    if initial_guess is None:
        current_result = xp.ones(signal.size)
        updated_result = xp.ones(signal.size)
    else:
        current_result = xp.copy(initial_guess)
        updated_result = xp.copy(initial_guess)

    signal_temporary = xp.copy(signal)
    kern_temporary = xp.copy(kern)

    if type(iteration_list) is int:
        iteration_list = [
            iteration_list,
        ]

    error = xp.zeros(iteration_list[-1] + 1)
    error[0] = xp.sum(
        (signal_temporary - fftconvolve(current_result, kern_temporary, "same")) ** 2
    )

    result = xp.zeros([signal.size, len(iteration_list)])

    inverse_kern = kern_temporary[::-1]
    signal_convolution_inverse_kern = fftconvolve(
        signal_temporary, inverse_kern, "same"
    )
    kern_convolution_inverse_kern = fftconvolve(kern_temporary, inverse_kern, "same")

    # To ensure we have non-negative iterations we apply a condition which is dependent on signal_convolution_inverse_kern
    # If signal_convolution_inverse_kern is negative then b = cutoff - scale_factor*np.amin(signal_convolution_inverse_kern).
    # Cutoff avoids division by 0.
    # If signal_convolution_inverse_kern is positive, let b = cutoff to ensure non-zero division.
    if xp.amin(signal_convolution_inverse_kern) < 0:
        b_min = xp.amin(signal_convolution_inverse_kern)
        b = cutoff - scale_factor * b_min
    else:
        b = cutoff

    count = 0

    for i in tqdm(range(1, iteration_list[-1] + 1), position=0, leave=True):
        # If an element in the previous iteration is very close to zero,
        # the same element in the next iteration should be as well.
        # This is handeled numerically by setting all elements <= cutoff to 0
        # and only performing the interation on those elements > cutoff.

        updated_convolution = fftconvolve(
            current_result, kern_convolution_inverse_kern, "same"
        )

        updated_result = (current_result * (signal_convolution_inverse_kern + b)) / (
            updated_convolution + b
        )

        error[i] = xp.sum(
            (signal_temporary - fftconvolve(updated_result, kern_temporary, "same"))
            ** 2
        )

        current_result[:] = updated_result[:]

        if i == iteration_list[count]:
            print(f"i = {iteration_list[count]}", flush=True)
            result[:, count] = updated_result[:]
            count += 1

    return result, error


def three_point_maxima(deconv_result, time_base=None, **kwargs):
    """
    Find amplitudes and arrival times of the deconvolved signal
    using scipy.signal.find_peaks.
    
    Use: Estimates arrival times and amplitudes of the FPP from the deconvolved signal.
    Input:
        deconv_result: result of deconvolution ............... numpy array
        time_base: (optional) time array ................ numpy array
        **kwargs ................................. passed to find_peaks
    Output:
        estimated_arrival_times: estimated location of arrivals ....... numpy array
        estimated_amplitudes: estimated amplitudes ................ numpy array
        
    If time_base is given, estimated_arrival_times are the arrival times in time_base. 
    If time_base is not given, ta are the peak locations in deconv_result.
    
    By default, this is a pure 3-point maxima. 
    In the presence of noise, we suggest using one of the following
    keywords in order to select for the peaks:
        height: required height of peak
        prominence: required prominence of peak
            may require setting wlen as well, for large arrays.
    
    In order to take the entire mass of each peak of deconv_result into account,
    the amplitudes are estimated by summing from one minima between two peaks
    to the minima between the next two peaks. The value of the minima is 
    divided proportionally between the two peaks, according to their height.
    ---min---peak----min----peak----min---
    ---][--sum range-][--sum range--][----
    """
    import numpy as np
    from scipy.signal import find_peaks

    # Find_peaks discounts the endpoints
    deconv_result_temporary = np.zeros(deconv_result.size + 2)
    deconv_result_temporary[1:-1] = deconv_result[:]

    peak_location = find_peaks(deconv_result_temporary, **kwargs)[0]

    estimated_amplitudes = deconv_result_temporary[peak_location]

    if time_base is None:
        return peak_location - 1, estimated_amplitudes
    else:
        return time_base[peak_location - 1], estimated_amplitudes
