import numpy as np
from scipy.stats import gamma, rv_continuous
import fppanalysis.correlation_function as cf
from fppanalysis.conditional_averaging import cond_av
from fppanalysis.running_moments import run_norm, run_norm_window
import matplotlib.pyplot as plt
from scipy.stats import uniform, norm
from scipy.signal import fftconvolve


def get_average(params: np.ndarray, distribution: rv_continuous):
    dist_scale = params[0]
    if distribution == uniform:
        return params[0]
    if distribution == norm:
        return params[0]
    if distribution.numargs > 0:
        return distribution.stats(params[1], scale=dist_scale, loc=0, moments="m")
    else:
        return distribution.stats(scale=dist_scale, loc=0, moments="m")


def get_pdf(params: np.ndarray, times: np.ndarray, distribution: rv_continuous):
    if distribution == uniform:
        pdf = np.zeros(len(times))
        shape = 1 / (1 + params[1] ** 2)
        low = params[0] * (1 - shape)
        high = params[0] * (1 + shape)
        pdf[np.logical_and(times < high, times >= low)] = 1 / (high - low)
        return pdf
    if distribution == norm:
        return distribution.pdf(times, loc=params[0], scale=params[1])
    if distribution.numargs > 0:
        pdf = distribution.pdf(times, params[1], loc=0, scale=params[0])
        if distribution == gamma and params[1] < 1:
            pdf[times == 0] = 0
        return pdf
    else:
        return distribution.pdf(times, loc=0, scale=params[0])


def plot_optimization(
    axe: plt.axis,
    times: np.ndarray,
    est_ccf: np.ndarray,
    est_acf: np.ndarray,
    params: np.ndarray,
    distribution: rv_continuous,
    xmin: float,
    xmax: float,
):
    axe.plot(
        times,
        est_ccf,
        label=r"$\widehat{R_{\tilde{\Phi}, \tilde{\Psi}}}(r)$",
        color="blue",
    )
    convo = fftconvolve(est_acf, get_pdf(params, times, distribution), "same")
    convo /= max(convo)
    axe.plot(
        times,
        convo,
        label=r"$\left<\widehat{\rho_\phi} \left( \frac{ r-d } {\tau} \right)\right>_d$",
        color="red",
    )

    axe.legend()
    axe.grid(True)
    axe.set_xlim(xmin, xmax)
    axe.set_ylim(0, 1.2)

    axe.set_xlabel(r"$r$")


def plot_pdf_function(
    axe: plt.axis,
    distribution: rv_continuous,
    error: float,
    parameters: np.ndarray,
    xmin: float,
    xmax: float,
    times: np.ndarray,
):
    axe.set_title("Distribution: {} Error {:.2g}".format(distribution.name, error))
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes

    axin = inset_axes(
        axe,
        width="20%",  # width = 30% of parent_bbox
        height=0.8,  # height : 1 inch
        loc=4,
    )
    axin.plot(times, get_pdf(parameters, times, distribution))
    axin.set_xlim(xmin, xmax)
    axin.set_title("Params: {}".format(parameters), fontsize=8)


def estimate_delays(
    x: np.ndarray,
    y: np.ndarray,
    dt: float,
    distribution: rv_continuous = gamma,
    ax: plt.axis = None,
    plot_pdf: bool = True,
    min_cutoff: float = 0,
):
    """
    Use:
       estimate_delays(x, y, dt, distribution=gamma, ax=None, plot="full", min_cutoff=0)

    Estimates the time delay distribution parameters for the pulses propagating from two measurement points.
    This is done by optimizing the time delay distribution parameters such that the predicted cross-correlation,
    given by the convolution of the autocorrelation and the time delay distribution, best fits the observed cross-
    correlation.

    Time series x, y, corresponding to each point measurement must be provided. The time delay distribution
    is assumed to follow a family distribution given by the argument distribution. Plots relevant autocorrelation
    and cross-correlation functions if a matplotlib ax is provided.

    Input:
        x: Time series ...................... (N,) np array
        y: Time series ...................... (N,) np array
        distribution: Assumed distribution .... class implementing scipy.stats.rv_continuous
        ax: Optional, if a matplotlib.pyplot.axis is provided, relevant plots will be plotted. These are meant to help
        understand the underlying principles of the optimization, the plots are not suitable for scientific publication
        .
        plot_pdf: Bool, if True, adds an inset plot of the time delay distribution function.
        min_cutoff: An upper bound for the cross-correlation maxima. Usage: If working with short time series
        or noisy data, it can be helpful for the method to set this value to an upper bound for the time that
        maximizes the cross-correlation, that is, a time such that you are sure that the cross-correlation is
        maximized before it.
    Output:
        avg: Average delay time
        params: Array of parameters that optimizes the time delay distribution function. In general params[0] will
        be the scale of the distribution, and params[1], if present, will give the shape of the distribution.
    """
    from scipy.optimize import minimize

    # Initialize distribution parameters. Uniform and norm require two parameters.
    parameters = (
        [1, 1] if distribution in [uniform, norm] else np.ones(1 + distribution.numargs)
    )

    ccf_times, est_ccf = cf.corr_fun(x, y, dt=dt, biased=False)
    _, est_acf = cf.corr_fun(x, x, dt=dt, biased=False)

    # We cut the cross correlation in the middle half to avoid noise near the ends
    est_ccf = est_ccf[np.abs(ccf_times) < max(ccf_times) / 2]
    est_acf = est_acf[np.abs(ccf_times) < max(ccf_times) / 2]
    ccf_times = ccf_times[np.abs(ccf_times) < max(ccf_times) / 2]

    max_cross_corr = max(ccf_times[np.argmax(est_ccf)], min_cutoff)
    parameters[0] = max_cross_corr
    # Time domain does not need to be full signal. This seems to be a good compromise.
    domain_cutoff = max_cross_corr * 100

    domain = np.abs(ccf_times) < domain_cutoff
    ccf_times = ccf_times[domain]
    est_ccf = est_ccf[domain]
    est_acf = est_acf[domain]
    est_ccf = est_ccf / max(est_ccf)

    def get_error(params):
        pdf = get_pdf(params, ccf_times, distribution)
        res = fftconvolve(est_acf, pdf, "same") * dt
        res /= max(res)
        return np.sum((res - est_ccf) ** 2)

    minimization = minimize(
        get_error, parameters, method="Nelder-Mead", options={"maxiter": 10000}
    )
    if not minimization.success:
        print("Optimization failed!!!")

    avg = get_average(minimization.x, distribution)

    if ax is not None:
        plot_optimization(
            ax,
            ccf_times,
            est_ccf,
            est_acf,
            minimization.x,
            distribution,
            xmin=-10 * max_cross_corr,
            xmax=20 * max_cross_corr,
        )
        if plot_pdf:
            plot_pdf_function(
                ax,
                distribution,
                minimization.fun,
                minimization.x,
                xmin=-max_cross_corr,
                xmax=5 * max_cross_corr,
                times=ccf_times,
            )

    return avg, minimization.x


def estimate_time_delay_ccmax(x: np.ndarray, y: np.ndarray, dt: float):
    """
    Estimates the average time delay between to signals by finding the time lag that maximizies the
    cross-correlation function.
    Returns:
        td Estimated time delay
        C Cross correlation at a time lag td.
    """
    ccf_times, ccf = cf.corr_fun(x, y, dt=dt, biased=True, norm=True)
    ccf = ccf[np.abs(ccf_times) < max(ccf_times) / 2]
    ccf_times = ccf_times[np.abs(ccf_times) < max(ccf_times) / 2]
    max_index = np.argmax(ccf)
    return ccf_times[max_index], ccf[max_index]


def estimate_time_delay_ccond_av_max(x: np.ndarray, x_t: np.ndarray, y: np.ndarray, y_t: np.ndarray):
    """
    Estimates the average time delay by finding the time lag that maximizies the
    cross conditional average of signal x when signal y is larger than threshold. 
    
    Input: 
        x: Signal to be conditionally averaged
        y: Reference signal 
        x_t: Time of signal x
        y_t: Time of signal y    

    Returns:
        td: Estimated time delay
        C: Cross conditional variance at a time lag td.
        events: Number of events larger than 2.5 the mean value
    """

    # Find length of time window for running normalization for both signals
    freq = 1e3
    windowx = run_norm_window(freq, x_t)  
    windowy = run_norm_window(freq, y_t)

    # Normalize signal
    signalx_norm, signalx_time_norm = run_norm(x, windowx, x_t)
    signaly_norm, signaly_time_norm = run_norm(y, windowy, y_t)

    # Cross conditional average
    threshold = 2.5
    Svals, s_av, s_var, t_av, peaks, wait = cond_av(
        signalx_norm, signalx_time_norm, threshold, Sref=signaly_norm
    )
    # Normalize conditional average waveform
    norm_s_av = s_av - min(s_av)
    norm_s_av = norm_s_av / max(norm_s_av)

    # Index of maximum normalized correlation value
    max_index = np.argmax(norm_s_av)

    return t_av[max_index], s_var[max_index], len(peaks)


def get_avg_velocity_from_time_delays(
    separation: float, params: np.ndarray, distribution: rv_continuous
):
    """
    Computes the average velocity corresponding to a given time delay distribution
    Input:
        separation: spatial separation between measurement points
        params: array of distribution parameters describing the time delay distribution
        distribution: Assumed time delay distribution .... class implementing scipy.stats.rv_continuous
    Output:
        Average velocity

    """
    from scipy.integrate import quad

    return quad(
        lambda td: separation / td * get_pdf(params, td, distribution), 0, np.infty
    )[0]


def get_velocity_pdf_from_time_delays(
    separation: float,
    velocities: np.ndarray,
    params: np.ndarray,
    distribution: rv_continuous,
):
    """
    Use:
       get_velocity_pdf_from_time_delays(separation, velocities, params, distribution)

    Returns the velocity probability density function given the time delay distribution described by the argument
    distribution with parameters params.

    Input:
        separation: spatial separation between measurement points
        velocities: array of velocities upon which the probability density function is evaluated
        params: array of distribution parameters describing the time delay distribution
        distribution: Assumed time delay distribution .... class implementing scipy.stats.rv_continuous
    Output:
        array with the probability density function of the velocities.
    """
    assert np.all(velocities > 0), "Velocities should be positive"
    return (
        get_pdf(params, separation / velocities, distribution)
        * separation
        / (velocities**2)
    )
