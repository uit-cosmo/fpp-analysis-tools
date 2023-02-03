import numpy as np


def rand_phase(p):
    # This assumes -pi<p<pi.
    # Move p to 0<p<2pi and add random phase
    pf = p + np.pi + np.random.uniform(0, 2 * np.pi, p.size)
    # Move phases back into 0<pf<2pi
    pf[pf > 2 * np.pi] -= 2 * np.pi
    # Move pf to -pi<pf<pi
    return pf - np.pi


def signal_rand_phase(S: np.ndarray):
    """
    Randomize the phases of a signal S,
    returning a signal with the same power spectral density
    but (close to) normally distributed signal values.

    Silently assumes a real signal S.
    """

    F = np.fft.rfft(S)

    pf = rand_phase(np.angle(F))
    # See np.fft.rfft: The zero frequency must be real,
    # as must the last freqency if there is an even number
    # of samples.
    pf[0] = 0
    if not S.size%2:
        pf[-1] = 0
    Ff = np.abs(F) * np.exp(1.0j * pf)
    
    # Specify length of inverse transform to correctly deal with odd samples
    return np.fft.irfft(Ff,len(S))
