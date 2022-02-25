"""
This file contains three methods for estimating the scaling of a time series.
resc_range and dfa compute  the rescaled range, and should scale as the
Hurst parameter.
wavelet_transform computes the power contained in scales, and should scale as
the negative slope of the power spectral density.
"""


def resc_range(X):
    """
    Use: RS,N = resc_range(X)
    This function calculates the rescaled range R(n)/S(n) of X.
    R/S method found at
    https://en.wikipedia.org/wiki/Hurst_exponent
    Input:
        X: Time series ................................... np array
    Output:
        RS: Rescaled range R(n)/S(n) ..................... (1xL) np array
        N: The sequence of points n. ....... Defaults to [L/8,L/16,L/32,...,8]
    """
    import numpy as np

    # Truncate X to Log2-size.
    L = int(np.floor(np.log2(X.size)))

    # Sizes of array to be split into.
    # To avoid bad statistics, we start at 8 series of length 2**L/8
    # and end at 2**L/8 time series of length 8.
    # This means that N contains the sizes of the time series
    # and N[::-1] contains the number of time series of that size.
    N = 2 ** np.arange(3, L - 2)

    # Calculate RS
    RS = np.zeros(N.size)

    for i in range(N.size):
        Xtemp = np.resize(X, (N[::-1][i], N[i]))
        M = np.resize(np.mean(Xtemp, axis=1), (N[::-1][i], 1))
        Z = np.cumsum(Xtemp - M, axis=1)
        R = np.max(Z, axis=1) - np.min(Z, axis=1)
        S = np.std(Xtemp, axis=1)
        RS[i] = np.mean(R / S)

    return RS, N


def dfa(X):
    """
    Use: Fn, N = dfa(X)
    This function calculates the rescaled range R(n)/S(n) of X
    by detrended fluctuation analysis.
    Method found at
    https://en.wikipedia.org/wiki/Detrended_fluctuation_analysis
    Input:
        X: Time series ................................... np array
    Output:
        Fn: Rescaled range R(n)/S(n) ..................... (1xL) np array
        N: The sequence of points n. ....... Defaults to [L/8,L/16,L/32,...,8]
    """
    import numpy as np

    # Truncate X to Log2-size.
    L = int(np.floor(np.log2(X.size)))

    # Sizes of array to be split into.
    # To avoid bad statistics, we start at 8 series of length 2**L/8
    # and end at 2**L/8 time series of length 8.
    # This means that N contains the sizes of the time series
    # and N[::-1] contains the number of time series of that size.
    N = 2 ** np.arange(3, L - 2)

    X = np.cumsum(X - X.mean())

    # Calculate F(n)
    Fn = np.zeros(N.size)

    for i in range(N.size):
        Xtemp = np.resize(X, (N[::-1][i], N[i]))
        Y = np.zeros((N[::-1][i], N[i]))
        M = np.arange(N[::-1][i])
        for n in range(N[i]):
            p = np.polyfit(M, Xtemp[:, n], 1)
            Y[:, n] = p[0] * M + p[1]

        Fn[i] = np.mean(np.std(Xtemp - Y, axis=1))

    return Fn, N


def wavelet_variance(X, dt=1, wavelet="haar"):
    """
    Use: wavelet_variance, scales = wavelet_transform(X, dt=1, wavelet='haar')
    This function calculates the wavelet variance at each scale, as described
    in [1].

    Input:
        X: Time series ................................... np array
        dt: Time step for the discretization ............. float
        wavelet: Wavelet to use, see [2] ................. string
    Output:
        wvar: The wavelet variance ....................... np array
        scales: The sequence of scales. .................. np array

    [1] Rypdal M. and Rypdal K., Earth Syst. Dynam. 7, 281-293 (2016)
    [2] http://wavelets.pybytes.com/
    """
    import numpy as np
    import pywt

    # Calculate scales
    L = pywt.dwt_max_level(X.size, wavelet)
    scales = dt * 2 ** np.arange(1, L + 1)

    # Calculate wavelet variance.
    # We only use the detail coefficients at each scale.
    coeffs = pywt.wavedec(X, wavelet)[1:][::-1]
    wvar = np.zeros(L)
    for i in range(L):
        wvar[i] = np.mean(np.abs(coeffs[i]) ** 2)

    return wvar, scales
