import numpy as np

"""
==============
peak_detection
=============

A set of functions used for peak detection in time series

detect_peaks_1d:(timeseries, delta_peak, threshold, peak_width=5)

"""


def detect_peaks_1d(timeseries, delta_peak, threshold, peak_width=5):
    """
    detect_peaks_1d

    Starting from the largest burst event in the time series at hand, we identify a set of
    disjunct sub records, placed symmetrically around the peak of burst events which exceed
    a given amplitude threshold until no more burst events exceeding this threshold are
    left uncovered.

    Used in Kube et al. PPCF 58, 054001 (2016).

    Input:
    ========
    timeseries.....ndarray, float: Timeseries to scan for peaks
    delta_peak.....integer: Separation of peaks in sampling points
    threshold......integer: Threshold a peak has to exceeed
    peak_width.....integer: Number of neighbouring elements a peak has to exceed

    Output:
    ========
    peak_idx_list...ndarray, integer: Indices of peaks in timeseries
    """

    # Sort time series by magnitude.
    max_idx = np.squeeze(timeseries.argsort())[::-1]

    # Remove peaks within delta_peak to the array boundary
    max_idx = max_idx[max_idx > delta_peak]
    max_idx = max_idx[max_idx < np.size(timeseries) - delta_peak]

    max_values = np.zeros_like(timeseries[max_idx])
    max_values[:] = np.squeeze(timeseries[max_idx])

    # Number of peaks exceeding threshold
    num_big_ones = np.sum(timeseries > threshold)
    try:
        max_values = max_values[:num_big_ones]
        max_idx = max_idx[:num_big_ones]
    except:
        print("detect_peaks_1d: No peaks in the unmasked part of the array.")
        return np.array([])

    # Mark the indices we need to skip here
    max_idx_copy = np.zeros_like(max_idx)
    max_idx_copy[:] = max_idx

    # Eliminate values exceeding the threshold within delta_peak of another
    # for idx, mv in enumerate(max_values):
    # print 'iterating over %d peaks' % ( np.size(max_idx))
    for i, idx in enumerate(max_idx):
        current_idx = max_idx_copy[i]
        if max_idx_copy[i] == -1:
            #    print 'idx %d is zeroed out' % (idx)
            continue

        # Check if this value is larger than the valueghbouring values of the
        # timeseries. If it is not, continue with next iteration of for loop
        if (
            timeseries[current_idx]
            < timeseries[current_idx - peak_width : current_idx + peak_width]
        ).any():
            max_idx_copy[i] = -1
            continue

        # Zero out all peaks closer than delta_peak
        close_idx = np.abs(max_idx_copy - idx)
        close_ones = np.squeeze(np.where(close_idx < delta_peak)[0])
        max_idx_copy[close_ones] = -1
        # Copy back current value
        max_idx_copy[i] = max_idx[i]

    # Remove all entries equal to -1
    max_idx_copy = max_idx_copy[max_idx_copy != -1]
    max_idx_copy = max_idx_copy[max_idx_copy < np.size(timeseries)]

    # Return an ndarray with all peaks of large amplitude indices
    return max_idx_copy
