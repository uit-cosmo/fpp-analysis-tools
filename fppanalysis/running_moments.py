def run_mean(S, radius):
    """
    Use:
        run_mean(S, radius)

    Computes the running average, using a method from
    https://stackoverflow.com/questions/14313510/how-to-calculate-moving-average-using-numpy

    Input:
        S: Signal to be averaged. ............ (N,) np array
        radius: Window size is 2*radius+1. ... int
    Output:

    """
    import numpy as np

    window = 2 * radius + 1
    rm = np.cumsum(S, dtype=float)
    rm[window:] = rm[window:] - rm[:-window]
    return rm[window - 1 :] / window


def run_moment(S, radius, moment=1, T=None):
    """
    Use:
        run_moment(S, radius, moment=1, T = None)

    Wrapper function for run_mean(), computes running mean and rms of S.
    To compute the running standard deviation of S, the running mean is
    subtracted from the signal.
    The running rms divides by window, not (window-1).

    Input:
        S: Signal to be averaged. ...................... (N,) np array
        radius: Window size is 2*radius+1. ............. int
        moment: Which running moment to compute. ....... int in [1,2,3,4]
                1: running mean.
                2: running standard deviation.
                3: running skewness
                4: running excess kurtosis
        T: Time base of S. ............................. (N,) np array
    Output:
        average: The running mean/rms of S, ............ (N-m*radius,)
                 depending on moment.                    np array
        signal: Signal with values not corresponding ... (N-m*radius,)
                to a running average removed.            np array
        time: time base corresponding to signal. ....... (N-m*radius,)
                                                         np array
        Here, m is 2 for the running mean and 4 for the other moments.
    """
    import numpy as np

    assert moment in range(1, 5)

    if moment == 1:
        if T is None:
            return run_mean(S, radius), S[radius:-radius]
        else:
            return run_mean(S, radius), S[radius:-radius], T[radius:-radius]

    elif moment == 2:
        rm = run_mean(S, radius)
        tmp = (S[radius:-radius] - rm) ** 2
        r_rms = np.sqrt(run_mean(tmp, radius))
        if T is None:
            return r_rms, S[2 * radius : -2 * radius]
        else:
            return r_rms, S[2 * radius : -2 * radius], T[2 * radius : -2 * radius]

    elif moment == 3:
        rm = run_mean(S, radius)
        tmp = S[radius:-radius] - rm

        r_skew = run_mean(tmp**3, radius) / run_mean(tmp**2, radius) ** 1.5
        if T is None:
            return r_skew, S[2 * radius : -2 * radius]
        else:
            return r_skew, S[2 * radius : -2 * radius], T[2 * radius : -2 * radius]

    elif moment == 4:
        rm = run_mean(S, radius)
        tmp = S[radius:-radius] - rm

        r_flat = run_mean(tmp**4, radius) / run_mean(tmp**2, radius) ** 2
        if T is None:
            return r_flat, S[2 * radius : -2 * radius]
        else:
            return r_flat, S[2 * radius : -2 * radius], T[2 * radius : -2 * radius]


def run_norm(S, radius, T=None, return_run_moment=False):
    """Performs the standard running normalization on S by subtracting a
    running mean and dividing by a running standard deviation.

    All outputs are returned with a size M=N-4*radius, corresponding to the
    running standard deviation.

    Input:
        S: Signal to be averaged. ....................... (N,) np array
        radius: Window size is 2*radius+1. .............. int
        T: Time base of S (optional). ................... (N,) np array
    Output:
        run_norm: S normalized, ......................... (M,) np array

        if T!=None:
            time: time base corresponding to Norm. ...... (M,) np array

        if return_run_moment==True:
            run_ave: Running mean of S. ................. (M,) np array
            run_std: Running standard deviation of S. ... (M,) np array
    """

    import numpy as np

    assert S.size > 2 * radius + 1, "Signal must be longer than window."

    run_ave, _ = run_moment(S, radius, moment=1)
    run_ave = run_ave[radius:-radius]

    run_std, _ = run_moment(S, radius, moment=2)

    run_norm = (S[2 * radius : -2 * radius] - run_ave) / run_std

    res = (run_norm,)

    if T is not None:
        res += (T[2 * radius : -2 * radius],)
    if return_run_moment:
        res += (
            run_ave,
            run_std,
        )

    # Drop the tuple if just the signal is to be returned.
    if len(res) == 1:
        res = res[0]

    return res


def window_radius(cut_off_freq, time):
    """Returns window radius used in running moments and normalization
    given a cut off frequency. Time step, dt, is computed from time.
    """
    import numpy as np

    dt = np.diff(time)[0]
    t_run_mean = 1 / cut_off_freq
    samples = ((t_run_mean / dt) - 1) / 2

    return round(samples)

def run_norm_ds(ds, radius):
    """Returns a running normalized xarray dataset of a given xarray dataset ds. The xarray function 
    'apply_ufunc' is using a running normalization function 'run_norm' from 
    https://github.com/uit-cosmo/fpp-analysis-tools/blob/main/fppanalysis/running_moments.py.
    
    Thanks to the Research software engineering (RSE) group at UiT (University of Tromsø) 
    for help with this.
        
    Input:
        - ds: xarray Dataset
        - radius: radius of the window used in run_norm. Window size is 2*radius+1. ... int
    
    Output:
        - ds_normalized: running normalized xarray Dataset

    'run_norm' returns a tuple of time base and the signal. Therefore, apply_ufunc will
    return a tuple of two DataArray (corresponding to time base and the signal).
    To return a format like the original dataset, we create a new dataset of normalized frames and
    corresponding time computed from apply_ufunc.

    Description of apply_ufunc arguments.
        - first the function
        - then arguments in the order expected by 'run_norm'
        - input_core_dimensions: list of lists, where the number of inner sequences must match
        the number of input arrays to the function 'run_norm'. Each inner sequence specifies along which
        dimension to align the corresponding input argument. That means, here we want to normalize
        frames along time, hence 'time'.
        - output_core_dimensions: list of lists, where the number of inner sequences must match
        the number of output arrays to the function 'run_norm'.
        - exclude_dims: dimensions allowed to change size. This must be set for some reason.
        - vectorize must be set to True in order to for run_norm to be applied on all pixels.
    """
    import xarray as xr
    
    normalization = xr.apply_ufunc(
        run_norm,
        ds["frames"],
        radius,
        ds["time"],
        input_core_dims=[["time"], [], ["time"]],
        output_core_dims=[["time"], ["time"]],
        exclude_dims=set(("time",)),
        vectorize=True,
    )

    ds_normalized = xr.Dataset(
        data_vars=dict(
            frames=(["y", "x", "time"], normalization[0].data),
        ),
        coords=dict(
            R=(["y", "x"], ds["R"].data),
            Z=(["y", "x"], ds["Z"].data),
            time=normalization[1].data[0, 0, :],
        ),
    )

    return ds_normalized

def run_mean_ds(ds, radius):
    """Returns a running mean xarray dataset of a given xarray dataset ds. The xarray function 
    'apply_ufunc' is using a running mean function 'run_mean' from 
    https://github.com/uit-cosmo/fpp-analysis-tools/blob/main/fppanalysis/running_moments.py.
    
    Thanks to the Research software engineering (RSE) group at UiT (University of Tromsø) 
    for help with this.
        
    Input:
        - ds: xarray Dataset
        - radius: radius of the window used in run_norm. Window size is 2*radius+1. ... int

    Output:
        - ds_mean: running mean xarray Dataset
    
    See run_norm_ds for description of apply_ufunc arguments.
    """
    import xarray as xr

    normalization = xr.apply_ufunc(
        run_mean,
        ds["frames"],
        radius,
        input_core_dims=[["time"], []],
        output_core_dims=[["time"]],
        exclude_dims=set(("time",)),
        vectorize=True,
    )

    ds_mean = xr.Dataset(
        data_vars=dict(
            frames=(["y", "x", "time"], normalization.data),
        ),
        coords=dict(
            R=(["y", "x"], ds["R"].data),
            Z=(["y", "x"], ds["Z"].data),
            time=ds["time"].values[radius:-radius],
        ),
    )
    return ds_mean