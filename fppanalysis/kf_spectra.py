import xarray as xr
import numpy as np
from scipy import signal


def get_kf_spectra_for_column(ds: xr.Dataset, column: int, sampling_time: float = 5e-7, spatial_spacing: float = 0.0055):
    """
    Computes the k-f spectrum for a given dataset at the given column.
    sampling_time is given in seconds, and spatial_spacing is given in cm.
    Returns:
        - k: wavenumbers
        - freqs: frequencies
        - s: Resulting spectra
    """
    data = ds.isel(x=column)['frames'].values
    data = np.nan_to_num(data)
    n = data.shape[1]

    # Apply Han window
    win = signal.get_window("han", n)
    dim_array = np.ones((1, data.ndim), int).ravel()
    dim_array[-1] = -1
    win = win.reshape(dim_array)
    data = win * data

    freqs = np.fft.rfftfreq(n, sampling_time)
    fft = np.fft.rfft(data, n, axis=1) / n

    k = np.fft.fftfreq(data.shape[0], d=spatial_spacing * 100)
    s = np.fft.fft(fft, n=data.shape[0], axis=0)/data.shape[0]

    fmean = np.nanmean(np.abs(fft), axis=0, keepdims=True)
    s /= fmean

    k = 2 * np.pi * np.fft.fftshift(k)
    s = np.abs(np.fft.fftshift(s, 0)) ** 2
    return k, freqs, s
