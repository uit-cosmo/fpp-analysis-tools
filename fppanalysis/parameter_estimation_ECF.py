"""
This file contains methods for parameter estimation from the empirical
characteristic function.
"""

import numpy as np
import scipy.optimize as sopt
import scipy.linalg as slg


def _empirical_characteristic_function(cf_var, data):
    """
    Use: _empirical_characteristic_function(cf_var, data)

    Returns the empirical characteristic function of data
    at cf_var.

    Input:
        cf_var: The variables for the characteristic function.
        data: array of observed values.

    """

    res = np.zeros([cf_var.size, 1], dtype=complex)
    for i in range(cf_var.size):
        res[i, 0] = np.mean(np.exp(1.0j * cf_var[i] * data))
    return res


def _error_vector(cf_var, data, CF, P, ECF=None):
    """
    Use: _error_vector(cf_var, data, ECF=None)

    Numerical estimate of the error variable n^(1/2) (ECF - CF(P))
    where n is the number of samples.

    Returns a vector containing all real parts of the error vector
    followed by all imaginary parts for compatability with the
    covariance estimate.
    """

    if ECF is None:
        ECF = _empirical_characteristic_function(cf_var, data)

    err = data.size ** (0.5) * (ECF - CF(cf_var, P))
    return np.vstack((np.real(err), np.imag(err)))


def _error_covariance_num(cf_var, data, ECF=None):
    """
    Use: _error_covariance_num(cf_var, data, ECF=None)

    Numerical estimate of the covariance of the
    error variable n^(1/2) (ECF - CF)
    """

    if ECF is None:
        ECF = _empirical_characteristic_function(cf_var, data)

    cf_var_v, data_v = np.meshgrid(cf_var, data)
    cov = np.exp(1.0j * cf_var_v * data_v) - ECF[:, 0]
    Crr = np.dot(np.real(cov).T, np.real(cov)) / (data.size - 1)
    Cri = np.dot(np.real(cov).T, np.imag(cov)) / (data.size - 1)
    Cii = np.dot(np.imag(cov).T, np.imag(cov)) / (data.size - 1)
    return np.block([[Crr, Cri], [Cri.T, Cii]])

def _error_covariance_an(u, P, CF):
    # Analytical covariance of the error variable n^(1/2) (ECF(u) - CF(u))
    # where n is the number of samples.
    Crr = np.zeros([u.size, u.size])
    Cri = np.zeros([u.size, u.size])
    Cii = np.zeros([u.size, u.size])

    for i in range(u.size):
        for j in range(u.size):
            Crr[i, j] = 0.5 * (
                np.real(CF(u[i] + u[j], P)) + np.real(CF(u[i] - u[j], P))
            ) - np.real(CF(u[i], P)) * np.real(CF(u[j], P))
            Cri[i, j] = 0.5 * (
                np.imag(CF(u[i] + u[j], P)) - np.imag(CF(u[i] - u[j], P))
            ) - np.real(CF(u[i], P)) * np.imag(CF(u[j], P))
            Cii[i, j] = 0.5 * (
                np.real(CF(u[i] - u[j], P)) - np.real(CF(u[i] + u[j], P))
            ) - np.imag(CF(u[i], P)) * np.imag(CF(u[j], P))
    return np.block([[Crr, Cri], [Cri.T, Cii]])

def est_from_ECF(data, CF, P0, cf_var_len=10, cf_var_step=1.0, **kwargs):
    import numpy as np

    """
    Use: est_from_ECF(data, CF, cf_var_len, cf_var_step, P0,
                      imthreshold=0.1, **kwargs)

    This function attempts paramteter estimation for a given
    characteristic function (CF) from a given data set.
    It assumes a sequence of N iid numbers
    X_n, n=1,2,....,N has been drawn from a distribution F(x,P),
    where P is a vector of parameters. The characteristic function of
    F(x,P) is CF(u,P) with u the characteristic function variable.

    It uses the method described in [1],[2] for iid data.

    Input:
        data: array of observed values .................... (Nx1) numpy array.
        CF: The characteristic function of X. ....... function, usage CF(u,P),
                                                      returning a complex
                                                      (u.size,1)-matrix.
        cf_var_len: Length of the variable array
                    for the characteristic function. ......... int, default 10
        cf_var_step: Step size of the variable. ............ float, default 1.
        P0: Initial guess for the parameter vector P. ............ numpy array

        **kwargs: keyword arguments passed to
                  scipy.optimize.minimize. ................. keyword arguments

    Output:
        Full result of scipy.optimize.minimize, see that function for details.

    Notes:
    * The default values for cf_var_len and cf_var_step are generally decent.
    * Longer cf_var_len is generally better, but takes longer to perform.
    * If cf_var_step is too small, the matrix is singular [1,3].
    * Some useful CFs are given in the file containing this method.

    References:
    [1] J. Yu, Econometric Reviews Vol. 23, pp. 93-123, 2004
    [2] K. C. Tran, Econometric Reviews Vol. 17, pp.167-183, 1998
    [3] M. Carrasco and J. P. florens,
        'Efficient GMM Estimation Using
        the Empirical Characteristic Function*', 2002, unpublished
    """

    cf_var = np.arange(1, cf_var_len + 1) * cf_var_step

    ECF = _empirical_characteristic_function(cf_var, data)

    CovMat = _error_covariance_num(cf_var, data, ECF=ECF)

    def minFun(P):
        # We are interested in minimizing errVec*CovMat^(-1)*np.conj(errVec)^T.
        # Instead of inverting CovMat, which is less accurate,
        # we solve the linear system
        # CovMat*x = np.conj(eps)
        # and do eps*x.
        eps = _error_vector(cf_var, data, CF, P, ECF=ECF)
        eps2 = slg.solve(CovMat, eps)
        res = np.dot(eps.T, eps2)[0, 0]

        return res

    return sopt.minimize(minFun, P0, **kwargs)


def asymptotic_covariance(cf_var, P, CF, CF_jac, samples, data=None, ECF=None):
    # Asymptotic covariance for the real case.
    # If data is provided, computes the numerical error covariance.
    # For samples>10**4,
    Jc = CF_jac(cf_var, P)
    Jr = np.real(Jc)
    Ji = np.imag(Jc)
    J = np.vstack((Jr, Ji))

    if data is None:
        C = _error_covariance_an(cf_var, P, CF)
    else:
        C = _error_covariance_num(cf_var, data, ECF)
    #print(f"Error covariance matrix condition number: {np.linalg.cond(C)}")
    tmp = slg.solve(C, J)
    tmp = np.dot(J.T, tmp)
    return slg.inv(tmp) / samples





def PDF_from_CF_fft(CF, P, Xlen=2**12, dX=0.01):
    """
    Use: X, pdf = PDF_from_CF_fft(CF, P, Xlen=2**12, dX=0.01)

    This function estimates the pdf of the random variable X by computing
    the inverse fourier transform of a characteristic function.

    Input:
        CF: The characteristic function of X. ....... function, usage CF(u,P),
                                                      returning a complex
                                                      (u.size,1)-matrix.
        P: List of parameters. ...................................... list
        Xlen: Number of data points in the random varaible array. ... int
        dX: step size of the random varaible array. ................. float

    Output:
        X: random variable array. ................................... np array
        pdf: array of pdf values. ................................... np array
    """
    import numpy as np

    Xfreq = np.fft.rfftfreq(Xlen, dX)
    u = 2 * np.pi * Xfreq

    pdf = np.fft.irfft(CF(u, P)[:, 0])[::-1] / dX
    pdf = np.fft.fftshift(pdf)

    X = np.arange(-Xlen / 2 + 1, Xlen / 2 + 1) * dX

    return X, pdf


def CF_exp(u, P):
    """
    Use: CF_exp(u, P)

    This function returns the characteristic function of an exponentially
    distributed variable as a complex (u.size, 1)-matrix.

    The parameter is the scale parameter beta, equal to the mean of the
    random variable.

    Input:
        u: The variable of the characteristic function. .......... 1D np array
        P=[beta, ]: The parameters of the exponential distribution. ...... list

    Output:
        res: the characteristic function. ........ complex (u.size,1) np array
    """
    import numpy as np

    res = np.zeros([u.size, 1], dtype=complex)
    res[:, 0] = (1.0 - 1.0j * P[0] * u) ** (-1.0)
    return res


def CF_jac_exp(u, P):
    # Jacobian matrix of the CF for the exponential distribution,
    # [dC/dP[0]]
    C = CF_exp(u, P)
    res = np.zeros([u.size, 1], dtype=complex)
    res[:, 0] = 1.0j * u * C[:, 0] ** 2
    return res


def CF_norm(u, P):
    res = np.zeros([u.size, 1], dtype=complex)
    res[:, 0] = np.exp(1.0j * P[0] * u - 0.5 * P[1] ** 2 * u**2)
    return res


def CF_jac_norm(u, P):
    C = CF_norm(u, P)
    res = np.zeros([u.size, 2], dtype=complex)
    res[:, 0] = 1.0j * u * C[:, 0]
    res[:, 1] = -P[1] * u**2 * C[:, 0]
    return res


def CF_gamma(u, P):
    """
    Use: CF_gamma(u, P)

    This function returns the characteristic function of a gamma
    distributed variable as a complex (u.size,1)-matrix.

    The parameters are the shape parameter k and the scale parameter theta.

    Input:
        u: The variable of the characteristic function. .......... 1D np array
        P=[k,theta]: The parameters of the gamma distribution. .......... list

    Output:
        res: the characteristic function. ........ complex (u.size,1) np array
    """
    import numpy as np

    res = np.zeros([u.size, 1], dtype=complex)
    res[:, 0] = (1.0 - 1.0j * P[1] * u) ** (-P[0])
    return res


def CF_jac_gamma(u, P):
    tmp = 1.0 - 1.0j * P[1] * u
    res = np.zeros([u.size, 2], dtype=complex)
    res[:, 0] = -(tmp ** (-P[0])) * np.log(tmp)
    res[:, 1] = 1.0j * P[0] * u * tmp ** (-1 - P[0])
    return res


def CF_gamma_norm(u, P):
    """
    Use: CF_gamma_norm(u, P)

    This function returns the characteristic function of a normalized gamma
    distributed variable as a complex (u.size, 1)-matrix.

    The gamma distributed random variable is normalized to have
    zero mean and unit rms.
    The parameter is the shape parameter k.

    Input:
        u: The variable of the characteristic function. .......... 1D np array
        P=[k, ]: The parameters of the gamma distribution. ............... list

    Output:
        res: the characteristic function. ........ complex (u.size,1) np array
    """
    import numpy as np

    res = np.zeros([u.size, 1], dtype=complex)
    res[:, 0] = np.exp(-1.0j * np.sqrt(P[0]) * u) * (
        1.0 - 1.0j * u / np.sqrt(P[0])
    ) ** (-P[0])
    return res
    
def CF_jac_gamma_norm(u, P):
    """
    Use: CF_jac_gamma_norm(u, P)

    This function returns the jacobian of the characteristic function 
    of a normalized gamma distributed variable as a complex (u.size, 1)-matrix.

    The gamma distributed random variable is normalized to have
    zero mean and unit rms.
    The parameter is the shape parameter k.

    Input:
        u: The variable of the characteristic function. .......... 1D np array
        P=[k, ]: The parameters of the gamma distribution. ............... list

    Output:
        res: the characteristic function. ........ complex (u.size,1) np array
    """
    import numpy as np

    C = CF_gamma_norm(u,P)
    res = np.zeros([u.size, 1], dtype=complex)
    v = 1.j*u/np.sqrt(P[0])
    res[:, 0] = -0.5*v*(2-v)/(1-v) - np.log(1-v)
    return res*C

def CF_gamma_gauss_norm(u, P):
    """
    Use: CF_gamma_gauss_norm(u, P)

    This function returns the characteristic function of Zn (defined below)
    as a complex (u.size, 1)-matrix.

    With random variables X ~ Gamma(k, theta), Y ~ Normal(0, sigma),
    Z = X + Y
    Zn = (Z-<Z>)/Z_rms.

    The parameters are:
    k, the shape parameter of the gamma distribution
    epsilon, defined as epsilon = Y_rms^2/X_rms^2 = sigma^2/(k*theta^2).

    Input:
        u: The variable of the characteristic function. .......... 1D np array
        P=[k, epsilon]: The parameters of the distribution. .............. list

    Output:
        res: the characteristic function. ........ complex (u.size,1) np array
    """
    import numpy as np

    res = np.zeros([u.size, 1], dtype=complex)

    v = 1.0j * u * (P[0] * (1 + P[1])) ** (-0.5)
    tmp1 = (1.0 - v) ** (-P[0])
    tmp2 = P[0] * (0.5 * P[1] * v**2 - v)
    res[:, 0] = tmp1 * np.exp(tmp2)

    return res


def CF_jac_gamma_gauss_norm(u, P):
    C = CF_gamma_gauss_norm(u, P)[:, 0]
    v = 1.0j * u * (P[0] * (1 + P[1])) ** (-0.5)

    res = np.zeros([u.size, 2], dtype=complex)
    res[:, 0] = -np.log(1 - v) - 0.5 * (v - 2) * v / (v - 1)
    res[:, 0] *= C
    res[:, 1] = 0.5 * P[0] * v**3 * C / ((1 + P[1]) * (v - 1))
    return res


def CF_general(u, P):
    """
    Use: CF_general(u, P)

    This function returns the characteristic function of Zn (defined below)
    as a complex (u.size, 1)-matrix.

    With X an FPP with exponential pulses and laplace distributed amplitudes,
    see Sec. 3.3 in [1], and  Y ~ Normal(0,sigma),
    Z = X + Y
    Zn = (Z-<Z>)/Z_rms.
    Zn has the characteristic function given by Eq. (45) in [1].

    The parameters are:
    gamma, the shape parameter of the distribution of X,
    beta, the shape parameter of the asymmetric laplace distribution
          of the amplitudes of X,
    epsilon, defined as epsilon = Y_rms^2/X_rms^2 = sigma^2/(k*theta^2).

    Input:
        u: The variable of the characteristic function. .......... 1D np array
        P=[gamma, beta, epsilon]: The parameters of the distribution. ... list

    Output:
        res: the characteristic function. ........ complex (u.size,1) np array

    References:
    [1] A. Theodorsen and O. E. Garcia, PPCF 60 (2018) 034006
        https://doi.org/10.1088/1361-6587/aa9f9c
    """
    import numpy as np

    res = np.zeros([u.size, 1], dtype=complex)

    B = np.sqrt((1.0 - P[1]) ** 3.0 + P[1] ** 3)
    v = 1.0j * u * (P[0] * (1 + P[2])) ** (-0.5) / B
    tmp1 = (1.0 + P[1] * v) ** (-P[0] * P[1])
    tmp2 = (1.0 - (1.0 - P[1]) * v) ** (-P[0] * (1.0 - P[1]))
    tmp3 = P[0] * (0.5 * P[2] * B**2.0 * v**2.0 - (1.0 - 2.0 * P[1]) * v)
    res[:, 0] = tmp1 * tmp2 * np.exp(tmp3)

    return res


def CF_general_lorentz(u, P):
    """
    Use: CF_general_lorentz(u, P)

    This function returns the characteristic function of Zn (defined below)
    as a complex (u.size, 1)-matrix.

    With X an FPP with lorentz pulses and laplace distributed amplitudes,
    see [1], and  Y ~ Normal(0,Y_rms),
    Z = X + Y
    Zn = (Z-<Z>)/Z_rms.
    Zn has the characteristic function given by Eq. (44) in [1].

    The parameters are:
    gamma, the shape parameter of the distribution of X,
    beta, the shape parameter of the asymmetric laplace distribution
          of the amplitudes of X,
    epsilon, defined as epsilon = Y_rms^2/X_rms^2.

    Input:
        u: The variable of the characteristic function. .......... 1D np array
        P=[gamma, beta, epsilon]: The parameters of the distribution. ... list

    Output:
        res: the characteristic function. ........ complex (u.size,1) np array

    References:
    [1] A. Theodorsen and O. E. Garcia, PPCF 60 (2018) 034006
        https://doi.org/10.1088/1361-6587/aa9f9c
    [2] O. E. Garcia and A. Theodorsen POP 25 (2018) 014506
        https://doi.org/10.1063/1.5020555
    """
    import numpy as np

    res = np.zeros([u.size, 1], dtype=complex)

    B2 = P[1] ** 3 + (1.0 - P[1]) ** 3
    gsq = np.sqrt(np.pi * P[0])
    v = u * ((1 + P[2]) * B2) ** (-0.5)

    tmp_e = -0.5 * P[2] * B2 * v**2
    tmp_mr = -1.0j * gsq * (1.0 - 2.0 * P[1]) * v
    tmp_LL1 = -1.0j * gsq * (P[1] ** 2) * v * (1 + 1.0j * P[1] * v / gsq) ** (-0.5)
    tmp_LL2 = (
        1.0j * gsq * ((1 - P[1]) ** 2) * v * (1 - 1.0j * (1 - P[1]) * v / gsq) ** (-0.5)
    )
    logres = tmp_e + tmp_mr + tmp_LL1 + tmp_LL2

    res[:, 0] = np.exp(logres)

    return res


def CF_bounded_Pareto(u, P):
    """
    Use: CF_bounded_Pareto(u, P)

    This function returns the characteristic function of an FPP with exponential
    pulses and bounded Pareto amplitudes [3] as a complex (u.size, 1)-matrix.

    The parameters are:
    gamma, the shape parameter of the distribution,
    alpha, the scale of the Pareto distribution,
    L, the lower bound of the Pareto distribution,
    H, the upper bound of the Pareto distribution.

    Input:
        u: The variable of the characteristic function. .......... 1D np array
        P=[gamma, alpha, L, H]: The parameters of the distribution. ... list

    Output:
        res: the characteristic function. ........ complex (u.size,1) np array

    References:
    [1] A. Theodorsen and O. E. Garcia, PPCF 60 (2018) 034006
        https://doi.org/10.1088/1361-6587/aa9f9c
    [2] O. E. Garcia and A. Theodorsen POP 25 (2018) 014506
        https://doi.org/10.1063/1.5020555
    [3] https://en.wikipedia.org/wiki/Pareto_distribution
    """
    import numpy as np
    import mpmath as mm

    # res = np.zeros([u.size, 1], dtype=complex)

    g_m = mm.mpf(P[0])
    a_m = mm.mpf(P[1])
    L_m = mm.mpf(P[2])
    H_m = mm.mpf(P[3])

    u_m = mm.matrix(-1.0j * u)
    C = mm.matrix(u.size, 1)

    def tmp(x, a):
        return -mm.log(x) - mm.gammainc(0, x) + x**a * mm.gammainc(-a, x)

    const_0 = -g_m * (a_m ** (-1) + mm.euler)
    const_1 = g_m / (H_m**a_m - L_m**a_m)

    for i in range(u.size):
        if u_m[i] == 0:
            lnC = 0
        else:
            lnCtmp = H_m**a_m * tmp(L_m * u_m[i], a_m) - L_m**a_m * tmp(
                H_m * u_m[i], a_m
            )
            lnC = const_0 + const_1 * lnCtmp
        C[i] = mm.exp(lnC)

    return np.array(C.tolist(), dtype=np.cfloat)
