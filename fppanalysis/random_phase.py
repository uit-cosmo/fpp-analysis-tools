import numpy as np


def rand_phase(p):
    # This assumes -pi<p<pi.
    # Move p to 0<p<2pi and add random phase
    pf = p + np.pi + np.random.uniform(0, 2 * np.pi, p.size)
    # Move phases back into 0<pf<2pi
    pf[pf > 2 * np.pi] -= 2 * np.pi
    # Move pf to -pi<pf<pi
    return pf - np.pi


def signal_rand_phase(S):
    # Returns a signal with the same power spectrum as S
    # but random phases.
    # Assumes S real.
    F = np.fft.rfft(S)

    pf = rand_phase(np.angle(F))
    Ff = np.abs(F) * np.exp(1.0j * pf)

    return np.fft.irfft(Ff)
