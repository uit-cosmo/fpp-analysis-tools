import warnings

import numpy as np
from scipy.stats import gamma, rv_continuous
import fppanalysis.correlation_function as cf
from fppanalysis.conditional_averaging import cond_av
from fppanalysis import utils
import matplotlib.pyplot as plt
from scipy.stats import uniform, norm
from scipy.signal import fftconvolve
from dataclasses import dataclass
from enum import Enum
from abc import ABC


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


class TDEOptions(ABC):
    pass


@dataclass
class CCFitOptions(TDEOptions):
    def __init__(
        self,
        fit_window=100,
        initial_guess=np.array([1, 0, 1]),
        interpolate: bool = False,
    ):
        """
        - fit_window: int The window employed for the fit will be centered at 0 with length 2 * fit_window + 1.
        - interpolate: If True the maximizing time lags are found by interpolation.
        """
        self.fit_window = fit_window
        self.initial_guess = initial_guess
        self.interpolate = interpolate

    @staticmethod
    def get_ccf_analytical(times, params):
        c, t0, taud = params[0], params[1], params[2]
        return c * np.exp(-np.abs(times - t0) / taud)


@dataclass
class ConditionalAvgOptions(TDEOptions):
    def __init__(
        self,
        min_threshold: float = 2.5,
        max_threshold: float = np.inf,
        delta: float = None,
        window: bool = False,
        interpolate: bool = False,
        verbose: bool = False,
    ):
        """
        - min_threshold: min threshold for conditional averaged events
        - max_threshold: max threshold for conditional averaged events
        - delta: If window = True, delta is the minimal distance between two peaks.
        - window: [bool] If True, delta also gives the minimal distance between peaks.
        - interpolate: If True the maximizing time lags are found by interpolation.
        - verbose: If True prints event info
        """
        self.min_threshold = min_threshold
        self.max_threshold = max_threshold
        self.delta = delta
        self.window = window
        self.interpolate = interpolate
        self.verbose = verbose


@dataclass
class CCOptions(TDEOptions):
    def __init__(self, interpolate: bool = False):
        """
        - interpolate: If True the maximizing time lags are found by interpolation.
        """
        self.interpolate = interpolate


class TDEMethod(Enum):
    CrossCorrelation = 1
    ConditionalAveraging = 2
    CrossCorrelationRM = 3
    CCFit = 4


class TDEDelegator:
    def __init__(self, method: TDEMethod, options, cache):
        self.method = method
        self.options = options
        self.cache = cache
        self.results = {}

    def estimate_time_delay(self, p1, p0, ds):
        if not self.cache:
            return self.estimate_time_delay_uncached(p1, p0, ds)

        saved = self._is_cached_or_reverse(p1, p0)
        if saved is None:
            new_result = self.estimate_time_delay_uncached(p1, p0, ds)
            self.results[hash((p1, p0))] = new_result
            return new_result
        return saved

    def _is_cached_or_reverse(self, p1, p0):
        hash_direct = hash((p1, p0))
        saved = self.results.get(hash_direct, None)
        if saved is not None:
            return saved

        hash_reverse = hash((p0, p1))
        saved = self.results.get(hash_reverse, None)
        if saved is not None:
            td, c, events = saved
            # If we already attempted to compute the td without success td will be None
            if td is None:
                return saved
            return -td, c, events

        return None

    def estimate_time_delay_uncached(self, p1, p0, ds):
        extra_debug_info = "Between pixels {} and {}".format(p1, p0)
        x = utils.get_signal(p1[0], p1[1], ds)
        y = utils.get_signal(p0[0], p0[1], ds)
        dt = utils.get_dt(ds)

        if utils.is_pixel_dead(x) or utils.is_pixel_dead(y):
            return None, None, None

        match self.method:
            case TDEMethod.CrossCorrelation:
                return estimate_time_delay_ccmax(
                    x, y, dt, self.options, extra_debug_info
                )
            case TDEMethod.ConditionalAveraging:
                return estimate_time_delay_ccond_av_max(
                    x, y, dt, self.options, extra_debug_info
                )
            case TDEMethod.CrossCorrelationRM:
                return estimate_time_delay_ccmax_running_mean(
                    x, y, dt, self.options, extra_debug_info
                )
            case TDEMethod.CCFit:
                return estimate_time_delay_ccf_fit(
                    x, y, dt, self.options, extra_debug_info
                )


def estimate_time_delay_ccf_fit(
    x: np.ndarray,
    y: np.ndarray,
    dt: float,
    estimation_options: CCFitOptions,
    extra_debug_info: str = "",
):
    """Estimates the average time delay between to signals by fitting the
    cross-correlation function to an analytical expression.

    Returns:
        td Estimated time delay
        C Cross correlation at a time lag td.
    """
    from scipy.optimize import minimize

    ccf_times, ccf = cf.corr_fun(x, y, dt=dt, biased=True, norm=True)
    ccf = ccf[np.abs(ccf_times) < max(ccf_times) / 2]
    ccf_times = ccf_times[np.abs(ccf_times) < max(ccf_times) / 2]
    max_index = np.argmax(ccf)
    max_time, ccf_value = ccf_times[max_index], ccf[max_index]

    fit_window = np.abs(ccf_times) < estimation_options.fit_window * dt

    minimization = minimize(
        lambda params: np.sum(
            (
                estimation_options.get_ccf_analytical(ccf_times[fit_window], params)
                - ccf[fit_window]
            )
            ** 2
        ),
        estimation_options.initial_guess,
        method="Nelder-Mead",
        options={"maxiter": 1000},
    )

    c, t0, taud = minimization.x[0], minimization.x[1], minimization.x[2]

    return t0, ccf_value, 0


def get_ccf_fit_data(
    x: np.ndarray, y: np.ndarray, dt: float, estimation_options: CCFitOptions
):
    """Used for debugging estimate_time_delay_ccf_fit.

    Returns:
        td Estimated time delay
        C Cross correlation at a time lag td.
    """
    from scipy.optimize import minimize

    ccf_times, ccf = cf.corr_fun(x, y, dt=dt, biased=True, norm=True)
    fit_window = np.abs(ccf_times) < estimation_options.fit_window * dt

    minimization = minimize(
        lambda params: np.sum(
            (
                estimation_options.get_ccf_analytical(ccf_times[fit_window], params)
                - ccf[fit_window]
            )
            ** 2
        ),
        estimation_options.initial_guess,
        method="Nelder-Mead",
        options={"maxiter": 1000},
    )

    return (
        ccf_times[fit_window],
        ccf[fit_window],
        estimation_options.get_ccf_analytical(ccf_times[fit_window], minimization.x),
    )


def estimate_time_delay_ccmax(
    x: np.ndarray,
    y: np.ndarray,
    dt: float,
    options: CCOptions,
    extra_debug_info: str = "",
):
    """Estimates the average time delay between to signals by finding the time
    lag that maximizes the cross-correlation function. If interpolate is True
    the maximizing lag is found by interpolation.

    Returns:
        td Estimated time delay
        C Cross correlation at a time lag td.
    """
    ccf_times, ccf = cf.corr_fun(x, y, dt=dt, biased=True, norm=True)
    ccf = ccf[np.abs(ccf_times) < max(ccf_times) / 2]
    ccf_times = ccf_times[np.abs(ccf_times) < max(ccf_times) / 2]
    max_index = np.argmax(ccf)
    max_time, ccf_value = ccf_times[max_index], ccf[max_index]
    if not options.interpolate:
        return max_time, ccf_value, 0

    # If the maximum is very close to the origin, we make an interpolation window of 20 discretization times in
    # each direction, otherwise, the interpolation window is twice the time maximum in each direction.
    interpolation_window_boundary = (
        20 * dt if np.abs(max_time) < 10 * dt else np.abs(max_time) * 2
    )
    interpolation_window = np.abs(ccf_times) < interpolation_window_boundary

    max_time_interpolate = _find_maximum_interpolate(
        ccf_times[interpolation_window], ccf[interpolation_window], extra_debug_info
    )

    return max_time_interpolate, ccf_value, 0


def estimate_time_delay_ccmax_running_mean(
    x: np.ndarray,
    y: np.ndarray,
    dt: float,
    options: CCOptions,
    extra_debug_info: str = "",
):
    """
    Estimates the average time delay between to signals by finding the time
    lag that maximizes the cross-correlation function. If the number of local maxima
    in the provided window is larger than 1, a running mean is applied on the estimated
    cross-correlation function with a running mean window of a size that will be
    increased gradually til the resulting cross-correlation function only has
    1 local maxima.
    If interpolate is True the maximizing lag is found by interpolation.
    Returns:
        td Estimated time delay
        ccf Cross-correlation value at the estimated time delay

    """
    ccf_times, ccf = cf.corr_fun(x, y, dt=dt, biased=True, norm=True)
    ccf = ccf[np.abs(ccf_times) < max(ccf_times) / 2]
    ccf_times = ccf_times[np.abs(ccf_times) < max(ccf_times) / 2]
    max_index = np.argmax(ccf)

    find_window = np.abs(ccf_times - ccf_times[max_index]) < 50 * dt
    ccf_times = ccf_times[find_window]
    ccf = ccf[find_window]

    ccf, n = _run_mean_and_locate_maxima(ccf)
    if ccf is None:
        warnings.warn("Maximum running window achieved " + extra_debug_info)
        return None, None, None
    if n > 1:
        ccf_times = ccf_times[int(n / 2) : -int(n / 2)]

    # Maximum might have changed after running mean
    max_index = np.argmax(ccf)
    max_time, ccf_value = ccf_times[max_index], ccf[max_index]

    if not options.interpolate:
        return max_time, ccf_value, 0

    interpolation_window = np.abs(ccf_times - max_time) < 20 * dt

    max_time_interpolate = _find_maximum_interpolate(
        ccf_times[interpolation_window], ccf[interpolation_window], extra_debug_info
    )

    return max_time_interpolate, ccf_value, 0


def get_time_delay_ccmax_rm_data(
    x: np.ndarray,
    y: np.ndarray,
    dt: float,
    interpolate: bool = False,
    extra_debug_info: str = "",
):
    """
    Returns all data relevant for the method estimate_time_delay_ccmax_running_mean.

    Returns:
        ccf_times Times for ccf array without running mean
        ccf cross correlation function array without running mean
        ccf_times_rm Times for ccf array with running mean
        ccf_rm ccf array with running mean
        td Estimated time delay
        max_ccf_value Max cross-correlation value
    """
    ccf_times, ccf = cf.corr_fun(x, y, dt=dt, biased=True, norm=True)
    ccf = ccf[np.abs(ccf_times) < max(ccf_times) / 2]
    ccf_times = ccf_times[np.abs(ccf_times) < max(ccf_times) / 2]
    max_index = np.argmax(ccf)

    find_window = np.abs(ccf_times - ccf_times[max_index]) < 50 * dt
    ccf_times = ccf_times[find_window]
    ccf = ccf[find_window]

    ccf_rm, n = _run_mean_and_locate_maxima(ccf)
    ccf_times_rm = ccf_times
    if ccf_rm is None:
        return None
    if n > 1:
        ccf_times_rm = ccf_times[int(n / 2) : -int(n / 2)]

    # Maximum might have changed after running mean
    max_index = np.argmax(ccf_rm)
    max_time, max_ccf_value = ccf_times_rm[max_index], ccf_rm[max_index]

    if not interpolate:
        return ccf_times, ccf, ccf_times_rm, ccf_rm, max_time, max_ccf_value

    interpolation_window = np.abs(ccf_times_rm - max_time) < 20 * dt

    max_time_interpolate = _find_maximum_interpolate(
        ccf_times_rm[interpolation_window],
        ccf_rm[interpolation_window],
        extra_debug_info,
    )

    return ccf_times, ccf, ccf_times_rm, ccf_rm, max_time_interpolate, max_ccf_value


def _run_mean_and_locate_maxima(ccf, max_run_window_size=7):
    ccf_mean = ccf
    n = 1
    while _count_local_maxima(ccf_mean) > 1:
        n = n + 2
        if n > max_run_window_size:
            return None, n
        ccf_mean = np.convolve(ccf, np.ones(n) / n, mode="valid")
    return ccf_mean, n


def _count_local_maxima(ccf):
    local_maxima = np.array([False] * len(ccf))
    local_maxima[1:-1] = np.logical_and(ccf[1:-1] > ccf[2:], ccf[1:-1] > ccf[:-2])
    return len(np.where(local_maxima)[0])


def _find_maximum_interpolate(x, y, extra_debug_info):
    from scipy.interpolate import InterpolatedUnivariateSpline

    # Taking the derivative and finding the roots only work if the spline degree is at least 4.
    spline = InterpolatedUnivariateSpline(x, y, k=4)
    possible_maxima = spline.derivative().roots()
    possible_maxima = np.append(
        possible_maxima, (x[0], x[-1])
    )  # also check the endpoints of the interval
    values = spline(possible_maxima)

    max_index = np.argmax(values)
    if possible_maxima[max_index] == x[0] or possible_maxima[max_index] == x[-1]:
        warnings.warn(
            "Maximization on interpolation yielded a maximum in the boundary!"
            + extra_debug_info
        )
    return possible_maxima[max_index]


def estimate_time_delay_ccond_av_max(
    x: np.ndarray,
    y: np.ndarray,
    dt: float,
    cond_av_eo: ConditionalAvgOptions,
    extra_debug_info: str = "",
):
    """Estimates the average time delay by finding the time lag that maximizes
    the cross conditional average of signal x when signal y is larger than
    threshold. Returns also the cross conditional variance at this maximum, and
    number of conditional averaged events.

    Input:
        x: Signal to be conditionally averaged
        x_t: Time base of signal x
        y: Reference signal
        min_threshold: min threshold for conditional averaged events
        max_threshold: max threshold for conditional averaged events
        delta: If window = True, delta is the minimal distance between two peaks.
        window: [bool] If True, delta also gives the minimal distance between peaks.
        interpolate: If True, interpolation is performed to find the maximum.

    Returns:
        float: Estimated time delay
        float: Cross conditional variance at a time lag td.
        int: Number of events larger than 2.5 the mean value
    """
    x_t = np.arange(0, dt * len(x), dt)

    _, s_av, s_var, t_av, peaks, _ = cond_av(
        x,
        x_t,
        smin=cond_av_eo.min_threshold,
        smax=cond_av_eo.max_threshold,
        Sref=y,
        delta=cond_av_eo.delta,
        window=cond_av_eo.window,
        print_verbose=False,
    )
    max_index = np.argmax(s_av)
    return_time = (
        _find_maximum_interpolate(t_av, s_av, extra_debug_info)
        if cond_av_eo.interpolate
        else t_av[max_index]
    )

    return return_time, s_var[max_index], len(peaks)


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
