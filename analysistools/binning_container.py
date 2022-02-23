import numpy as np


class binning_container(object):
    """
    Define the binning_container data type used for conditional averaging

    Example: (See also the example program at the end of the file)

    blob_bc = binning_container(num_bins, bin_length, bin_edges,
                                bin_function, mode)
    num_bins: Number of bins, equal to the number of conditional events
    bin_length: Length of each bin (each bin is a np.ndarray)
    bin_function: Function operating on each conditional window that
                  determines the bin in which the conditional window data
                  is put.
    mode: Either add or append. If add, the instances adds the
          argument interval to the interval in each bin
          If mode=='append', the instance appends the argument
          interval to the list for each bin

    When calling blob_bc( array ), call bin_function(array) to
    determine which bin it adds to.

    For example
    We want to bin 42 conditional events, each 20 samples long in bins where
    2.5 < max(event) < 3.5
    3.5 < max(event) < 4.5
    4.5 < max(event) < 5.5
    5.5 < max(event) < 100.0

    num_peaks = 42
    burst_length = 20

    First create the binning container object:
    blob_bc = binning_containter(num_peaks, burst_length,
                                 np.array([2.5, 3.5, 4.5, 5.5, 100.0])
                                 lambda x: x.max() )

    Assume that we know the center of the conditional events, f.ex. by
    peak_arr = peaks_ne = detect_peaks_1d(ts, burst_separation, burst_threshold)
    peak_arr is a np.array, containing the peaks in ts
    assert(peak_arr.shape[0] == num_peaks)

    Iterate over the peaks and put each conditional event the appropriate bin:
    for peak_idx, peak_tidx in enumerate(peak_arr):
        # Get a view on the current peak in ts
        ts_cut = ts[peak_tidx - 10:peak_tidx + 10]
        # This puts ts_cut into the correct bin
        blob_bc.bin(ts_cut)

    Get binned bursts with 3.5 < max(burst) < 4.5
    # blob_bc[1]
    This returns a np.array with dimension [n, burst_length],
    where n is the number of bursts within this bin.

    """

    def __init__(self, num_bins, bin_length, bin_edges, bin_function, mode="add"):
        """
        num_bins:......
        bin_length:....
        bin_edges:.....
        bin_function:..
        """
        assert mode in ["add", "append"]

        self.num_bins = num_bins
        self.bin_length = bin_length
        self.bin_edges = list(zip(bin_edges[:-1], bin_edges[1:]))

        self.bin_function = bin_function
        self.mode = mode

        self.bin_max = bin_edges.max()
        self.bin_min = bin_edges.min()
        # Create list of bins
        self.bins = []
        # Fill the bins
        for ibin in np.arange(num_bins):
            # If we add to the bins, insert an intervall we keep adding to
            if self.mode == "add":
                self.bins.append(np.zeros(bin_length, dtype="float64"))
            elif self.mode == "append":
                self.bins.append([])

        self.count = np.zeros(num_bins, dtype="int")

    def max(self, array):
        return array.max()

    def bin(self, array, feval_array=None):
        # Bin the data in array into the according bin
        # If supplied, use feval_array to determine the bin
        # array is binned into.
        # If feval_array == None, use array to determine the
        # bin used

        assert array.size == self.bin_length

        # Find the bin we bin ''array'' in
        if feval_array is None:
            # If feval_array is unspecified, pass ''array'' to bin_function
            rv = self.bin_function(array)
        else:
            # if feval_array is specified, pass ''feval_array'' to bin_function
            rv = self.bin_function(feval_array)

        # Perform boundary checks of rv against the upper and lower bin
        # boundary
        if rv > self.bin_max:
            raise ValueError("Could not bin array: %f > max(bin_edges)" % rv)

        if rv < self.bin_min:
            # raise ValueError('Could not bin array: %f < min(bin_edges)' % rv)
            raise ValueError("Could not bin array: %f < %f" % (rv, self.bin_min))

        idx = np.argwhere(
            np.array([(rv > t1) & (rv <= t2) for t1, t2 in self.bin_edges])
        )[0][0]
        # Add to the appropriate bin
        if self.mode == "add":
            (self.bins[idx])[:] = (self.bins[idx])[:] + array
        elif self.mode == "append":
            self.bins[idx].append(array)
        # Increase bin counter
        self.count[idx] = self.count[idx] + 1

    def count(self, bin_idx=None):
        # return the count of each bin
        if bin_idx is None:
            return self.count
        else:
            return self.count[bin_idx]

    def get_num_bins(self):
        return len(self.num_bins)

    def __getitem__(self, idx):
        if self.mode == "add":
            return self.bins[idx]
        elif self.mode == "append":
            return np.array(self.bins[idx])

    def condvar(self, idx):
        """
        Compute the conditional variance of the data stored in bin idx

        Input:
        ======
        idx.......int, gives the index to the bin we compute the cond. var from

        Output:
        =======
        cvar.....ndarray(float), the conditional variance at each sampling point
        """

        all_bursts = np.array(self.bins[idx])
        ca = all_bursts.mean(axis=0)
        tmp = all_bursts - ca[np.newaxis, :].repeat(all_bursts.shape[0], axis=0)
        cvar = 1.0 - (tmp ** 2.0).mean(axis=0) / (all_bursts ** 2.0).mean(axis=0)
        return cvar


# Exemplary use of binning_container to find conditional averages
if __name__ is "__main__()":
    # define the conditional averaging threshold
    burst_threshold = 2.5
    # define the separation between neighbouring bursts, in samples
    burst_separation = 250
    # define the length of a burst event, in samples
    burst_length = 250
    # define the bin boundaries in which we bin the bursts. This corresponds to
    # bursts 2.0 <= A < 4.0
    #        4.0 <= A < 6.0
    # 6.0 <= A 100.0 (100.0  is a maximum and may be the maximum of the time
    # series at hand
    burst_boundaries = np.array([2.0, 4.0, 6.0, 100.0])

    # Now, a get a time series
    ts = np.random.uniform(0.0, 5.0, 10000)

    # Normalize the time series. This will be our reference time series
    ts_ref = (ts - ts_mean()) / ts.std(ddof=1)

    # Lets say we also have another time series, for cross-conditional
    # averaging
    ts = np.random.uniform(0.0, 5.0, 1000)
    ts_x = (ts - ts.mean()) / ts_std(ddof=1)

    # Get the indices in the time series where a burst is detected
    ref_burst_idx = detect_peaks_1d(
        ts_norm, burst_separation, burst_threshold, peak_width=5
    )
    ref_num_bursts = ref_burst_idx.size
    print("Detected %d bursts in the signal" % (ref_num_bursts))

    # Now we run conditional averaging with the binning container
    # For this, we create a binning container that will store all
    # sub-intervals around the peaks that were detected in the time series
    burst_bc = binning_container(
        ref_num_bursts,
        2 * burst_length,
        burst_boundaries,
        lambda x: x.max(),
        mode="append",
    )

    binned_bursts = 0
    # Iterate over the bursts and pub each
    for b_idx, b_tidx in enumerate(ref_burst_idx):
        # b_idx is the index of the current burst in the array of all detected bursts
        # b_tidx is the index of the current burst in the time series at hand

        # First, we create a view on the intervall around the bin in the reference
        # signal
        ts_ref_cut = ts_norm[b_tidx - burst_length : b_tidx + burst_length]

        # Assume we have a another time series for cross-conditional averaging.
        # Let's create a view on this time series in the same intervall
        ts_x_cut = ts_x[b_tidx - burst_length : b_tidx + burst_length]

        # N
        try:
            # This will add the waveform ts_x_cut into the bin determined by
            # the max of the reference waveform
            burst_bc.bin(ts_x_cut, feval_array=ts_ref_cut)
            binned_bursts += 1
        except:
            # something went wrong.
            continue

    # Noe we can compute the conditionally averaged waveform in one
    # amplitude range, f.ex. 1, 4.0 <= A < 6.0
    all_bursts = burst_bc[1]
    num_bursts = len(burst_bc[1])

    # This is the conditionally averaged waveform
    ca = all_bursts.mean(axis=0)

    # Compute the conditional variance
    tmp = all_bursts - ca[np.newaxis, :].repeat(num_bursts, axis=0)
    cvar = 1.0 - (tmp ** 2.0).mean(axis=0) / (all_bursts ** 2.0).mean(axis=0)
