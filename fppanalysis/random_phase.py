import numpy as np
from typing import Optional

def signal_rand_phase(S: np.ndarray, seed: Optional[int] = None) -> np.ndarray:
    """
    Randomize the phases of a signal S,
    returning a signal with the same power spectral density
    but (close to) normally distributed signal values.

    Silently assumes a real signal S.
    """

    F = np.fft.rfft(S)

    rng = np.random.default_rng(seed)
    pf = rng.uniform(-np.pi,np.pi,size=F.size) 
    # See np.fft.rfft: The zero frequency must be real,
    # as must the last freqency 
    # if there is an even number of samples.
    pf[0] = 0
    if not S.size%2:
        pf[-1] = 0
    Ff = np.abs(F) * np.exp(1.0j * pf)
    
    # Specify length of inverse transform to correctly deal with odd samples
    return np.fft.irfft(Ff,len(S))
