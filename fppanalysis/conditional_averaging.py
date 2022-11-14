def cond_av(S, T, smin=None, smax=None, Sref=None, delta=None, window=False, 
            prominence=None, threshold = None, weight='amplitude'):
    """
    Use: cond_av(S, T, smin, smax=None, Sref=None, delta=None, window=False)
    Use the level crossing algorithm to compute the conditional average of
    a process.
    Inputs:
        S: Signal. Size N ................................. (1xN) np array
        T: Time base ...................................... (1xN) np array
        smin: Minimal peak amplitude
              in units of rms-value above mean value. ..... Either float, 
              None or (1xN) np array. def None
        smax: Maximal peak amplitude. ..................... Either float, 
        None or (1xN) np array. def None
        Sref: Reference signal.
              If None, S is the reference. ................ (1xN) np array,
                                                            def None
        delta: The size of the conditionally averaged signal. If window = True,
               it is also the minimal distance between two peaks.
               If delta = None, it is estimated as
               delta = len(S)/(number of conditional events)*timestep.
               ............................................ float, def None
        window: If True, delta also gives the minimal distance between peaks.
                ........................................... bool, def False
        prominence: Minimal peak prominence in units
                    of rms-value above mean value.......... Either a number, 
                    None, an array matching x or a 2-element sequence of the 
                    former. The first element is always interpreted as the 
                    minimal and the second, if supplied, as the maximal required 
                    prominence. def None
        threshold: Required threshold of peaks, the vertical distance to its 
                   neighboring samples. ................... Either a number, 
                   None, an array matching x or a 2-element sequence of the 
                   former. The first element is always interpreted as the 
                   minimal and the second, if supplied, as the maximal required
                   threshold.  def None
        weight: Weighting to be used in the conditionally averaged signal. If
                weight='amplitude' the amplitudes of each peak decides its 
                weight in the average. If weight='equal' all peaks are 
                weighted equally in the average............ str, def 'amplitude'
    Outputs:
        Svals: Signal values used in the conditional average.
               S with unused values set to nan. ........... (1xN) np array
        s_av: conditionally averaged signal ............... np array
        s_var: conditional variance of events ............. np array
        t_av: time base of s_av ........................... np array
        peaks: max amplitudes of conditionally averaged events
        wait: waiting times between peaks
    """
    import numpy as np
    from scipy.signal import find_peaks
    
    if all(i is None for i in[smin, prominence]):
        raise TypeError('Missing 1 required positional argument: \'smin\' '
                        'or \'prominence\'')
    
    if Sref is None:
        Sref = S
    assert len(Sref) == len(S) and len(S) == len(T)

    sgnl = (Sref - np.mean(Sref)) / np.std(Sref)
    dt = sum(np.diff(T)) / (len(T) - 1)
    
    #Estimating delta.
    if delta is None:
        if smin is None:
            tmpmin = prominence
        else:
            tmpmin = smin
            
        places = np.where(sgnl > tmpmin)[0]
        dplaces = np.diff(places)
        split = np.where(dplaces != 1)[0]
        # (+1 since dplaces is one ahead with respect to places)
        lT = np.split(places, split + 1)
        delta = len(sgnl) / len(lT) * dt
    
    distance = None
    # Ensure distance delta between peaks.
    if window:
        distance = int(delta / dt)
        
    if prominence is None:
        prominence = smin / 4
    
    # Find peak indices.
    gpl_array, _ = find_peaks(sgnl, height = [smin, smax], distance = distance,
                              prominence = prominence, threshold = threshold)
    
    # Use arange instead of linspace to guarantee 0 in the middle of the array.
    t_av = np.arange(-int(delta / (dt * 2)), int(delta / (dt * 2)) + 1) * dt


    peaks = S[gpl_array]
    wait = np.append(np.array([T[0]]), T[gpl_array])
    wait = np.diff(wait)

    Svals = np.zeros(len(sgnl))
    Svals[:] = np.nan

    badcount = 0

    t_half_len = int((len(t_av) - 1) / 2)
    s_tmp = np.zeros([len(t_av), len(gpl_array)])

    for i in range(len(gpl_array)):
        global_peak_loc = gpl_array[i]

        # Find the average values and their variance
        low_ind = int(max(0, global_peak_loc - t_half_len))
        high_ind = int(min(len(sgnl), global_peak_loc + t_half_len + 1))
        tmp_sn = S[low_ind:high_ind]
        Svals[low_ind:high_ind] = S[low_ind:high_ind]
        if low_ind == 0:
            tmp_sn = np.append(np.zeros(-global_peak_loc + t_half_len), tmp_sn)
        if high_ind == len(S):
            tmp_sn = np.append(
                tmp_sn, np.zeros(global_peak_loc + t_half_len + 1 - len(S))
            )
        if max(tmp_sn) != tmp_sn[t_half_len]:
            badcount += 1
        
        s_tmp[:, i] = tmp_sn
        if weight == 'equal':
            s_tmp[:, i] /= tmp_sn[t_half_len]
    s_av = np.mean(s_tmp, axis=1)

    # The conditional variance of the conditional event f(t) is defined as
    # CV = <(f-<f>)^2>/<f^2> = 1 - <f>^2/<f^2>
    # at each time t.
    # For a highly reproducible signal, f~<f> and CV = 0.
    # For a completely random signal, <f^2> >> <f>^2 and CV = 1.
    # OBS: We return 1-CV = <f>^2/<f^2>.
    s_var = s_av ** 2 / np.mean(s_tmp ** 2, axis=1)
    print("conditional events:{}".format(len(peaks)), flush=True)
    if badcount > 0:
        print("bursts where the recorded peak is not the largest:" + str(badcount))


    return Svals, s_av, s_var, t_av, peaks, wait
