import numpy as np


def excess_stat(S, A, dt, pdf=False, N=32):
    """
    For a given signal S and a given threshold values a in A, this function finds the excess statitics of S.

    Input:
        S: the signal, a 1d numpy array (1d np)
        A: the threshold values, 1d np
        dt: the time resolution, float
        pdf: if set to True, the function estimates the PDF of the time above threshold as well.
        N: The number of bins in the estimation of the PDF above threshold.
    Output:
        Theta_array: The total time above threshold for each value in A, 1d np.
        X_array: The total number of upwards crossings over the threshold for each value in A, 1d np.
        avT_array: The average time above threshold for each value in A, 1d np.
        rmsT_array: The rms-value of time above threshold for each value in A, 1d np.

        If pdf is set to True, we additionaly output
        Tpdf: A 2d numpy array, shape (A.size,N=32). For each value in A, this is the estimated PDF of time above threshold.
        t: Time values for Tpdf, same shape.
    """
    Theta_array = np.array([])
    X_array = np.array([])
    avT_array = np.array([])
    rmsT_array = np.array([])
    if pdf:
        dT_dict = {}
    for a in A:
        # This is the basis: the parts of the signal that are above the
        # threshold.
        places = np.where(S > a)[0]
        if len(places) > 0:
            # print('Num, places to check:{}'.format(len(places)))
            Theta = dt * len(places)

            # Find X, avT an distribution of dT
            # Each blob is connected, so discrete blobs have more than one time
            # length between them
            dplaces = places[1:] - places[:-1]
            split = np.where(dplaces != 1)[0]  # split the array
            # at places where distance between points is greater than one
            lT = np.split(dplaces, split)
            lT[0] = np.append(lT[0], 1)  # To get correct length of the first
            # Number of upwards crossings is equal to number of discrete blobs
            X = len(lT)
            if places[0] == 0:
                # Don't count the first blob if there is no crossing.
                X += -1
            dT = np.array(
                [dt * len(lT[i]) for i in range(0, len(lT))]
            )  # Array of excess times
            avT = np.mean(dT)
            rmsT = np.std(dT)
        elif len(places) == 0:
            Theta = 0
            X = 0
            avT = 0
            rmsT = 0
            dT = np.array([])
        Theta_array = np.append(Theta_array, Theta)
        X_array = np.append(X_array, X)
        avT_array = np.append(avT_array, avT)
        rmsT_array = np.append(rmsT_array, rmsT)
        if pdf:
            dT_dict.update({a: dT})

    if pdf:

        def Th_dT(dT, dt, A):
            """
            Calculate the pdf P(dT|A) and avT from this pdf.
            dT: dictionary. From above

            Returns the 2d time array t, and the 2d-array dTpdf, containing the pdfs.
            t and dTpdf are both 2d-arrays storing the values for each a along the axis. The pdf for A[i] is dTpdf[i,:], t[i,:].
            """
            dTpdf = np.zeros((len(A), N))
            t = np.zeros((len(A), N))

            for i in range(0, len(A)):
                a = A[i]
                if len(dT[a]) >= 1:
                    dTpdf[i, :], bin_edges = np.histogram(dT[a], bins=N, density=True)
                    t[i, :] = (bin_edges[1:] + bin_edges[:-1]) / 2  # Record bin centers
                else:
                    continue  # Need not do anything, everything is zeroes.
            return dTpdf, t

        dTpdf, t = Th_dT(dT_dict, dt, A)

        return Theta_array, X_array, avT_array, rmsT_array, dTpdf, t
    else:
        return Theta_array, X_array, avT_array, rmsT_array
