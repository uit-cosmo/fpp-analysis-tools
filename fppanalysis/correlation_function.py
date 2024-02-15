import numpy as np
import scipy.signal as ssi


def corr_fun(X, Y, dt, norm=True, biased=True, method="auto"):
    """
    Estimates the correlation function between X and Y using ssi.correlate.

    For consistency with ssi.correlate and np.correlate, the estimated correlation is
    R(tb) = <X(t+tb) Y(t)> = <X(t) Y(t-tb)>
    where R is the correlation function and tb is the returned correlation time.
    For consistency with the definition
    R'(tb) = <X(t) Y(t+tb)>
    the correlation function array must be inverted.
    
    For now, we require both signals to be of equal length.
    
    Input:
        X: First array to be correlated. ............................. (Nx1) np.array
        Y: Second array to be correlated. ............................ (Nx1) np.array
        dt: Time step of the time series. ............................ float
        norm: Normalizes the correlation function to a maxima of 1 ... bool
        biased: Trigger estimator biasing. ........................... bool
        method: 'direct', 'fft' or 'auto'. Passed to ssi.correlate ... string
    
    For biased=True, the result is divided by X.size.
    For biased=False, the estimator is unbiased and returns the result
    divided by X.size-|k|, where k is the lag.
    The unbiased estimator diverges for large lags, and
    for small lags and large X.size, the difference is trivial.
    """

    assert X.size == Y.size

    if norm:
        Xn = (X - X.mean()) / X.std()
        Yn = (Y - Y.mean()) / Y.std()
    else:
        Xn = X
        Yn = Y

    R = ssi.correlate(Xn, Yn, mode="full", method=method)

    k = np.arange(-(X.size - 1), X.size)
    tb = k * dt

    if biased:
        R /= X.size
    else:
        R /= X.size - np.abs(k)

    return tb, R
