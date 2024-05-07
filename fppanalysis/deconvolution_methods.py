"""
 Methods for performing Richardson-Lucy deconvolution.

 A kernel (pulse shape) is needed. If this is not known, it must be estimated.
 The duration time and asymmetry parameter of the two-sided exponential pulse may be estimated from the power spectral density of the signal.
 For best results, ensure the mass of the pulse shape is localized in the middle of the kernel array.

 The method requires a positive definite signal (only positive pulses). 
 If you have a normalized signal, you must undo the normalization before using the deconvolution.
 If signal is FPP+noise with intermittency parameter gamma and noise parameter epsilon, undo the normalization with
 signal = ((1+eps)*gamma)**(0.5)*normalized_signal + gamma
 This can be used in the RL_gauss_deconvolve function.
 Generally, more iterations are better.
 Check that the result falls to zero during quiet periods, or amplitudes may not be correctly calculated.

 Then, use three_point_maxima function to find the peaks.
 Noise is handled by setting a height threshold related to  the intermittency parameter (gamma), noise to signal ratio (epsilon), and the mean amplitude of the signal (<A>):
 threshold = <A>*square_root(gamma*epsilon)

 If you want to reconstruct the signal from the result of the deconvolution and compare this to the normalized data, normalize the reconstructed signal.
"""

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
        signal: Signal to be deconvolved ..................................... 1D np/cp array
                If gpu=True, this has to be a cupy array.
        kern: Deconvolution kernel, this can be the pulse shape or the forcing  1D np/cp array
               If gpu=True, this has to be a cupy array.
        iteration_list: The number of iterations. ............................. int or list of int
                        If this is a list, the deconvolution result is returned for each element in iteration_list, see below.
        initial_guess: Initial array guess. Leave blank for all zeros. ........ 1D np/cp array
        cutoff: For avoiding divide by zero errors. ........................... float
        scale_factor: Scale factor which is multiplied with b condition........ float > 0, default = 1
        gpu: Use GPU-accelerated version....................................... bool
    Output:
        result: result array. NxM, where N=len(signal) and M=len(iteration_list)   np/cp array
        error: normalized mean difference between iterations .................... 1D np/cp array

    WARNING:
    For estimating the pulse shape, you need to ensure you have an odd number of data points when generating synthetic data.
    Do a check like the following before putting signal array into the 'signal' argument.
    if (len(signal) % 2) == 0:
        signal = signal[:-1]
        time = time[:-1]
    """
    from tqdm import tqdm

    if gpu:
        import cupy as xp
        from cusignal.convolution.convolve import fftconvolve
    else:
        import numpy as xp
        from scipy.signal import fftconvolve

    if gpu:
        pool = xp.cuda.MemoryPool(xp.cuda.malloc_managed)
        xp.cuda.set_allocator(pool.malloc)

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
        ) / len(signal)

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
                If time_base is given, estimated_arrival_times are the arrival times in time_base. 
                If time_base is not given, estimated_arrival_times are the peak locations in deconv_result.
        **kwargs: Keyword arguments are passed to scipy.find_peaks function
                By default, the pure 3-point maxima uses the scipy.find_peaks function.
                In the presence of noise, we suggest using one of the following
                keyword arguments in order to select for the peaks:
                    height: required height of peak
                    prominence: required prominence of peak
                        may require setting wlen as well, for large arrays.
    Output:
        estimated_arrival_times: estimated location of arrivals ....... numpy array
        estimated_amplitudes: estimated amplitudes ................ numpy array
        
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
