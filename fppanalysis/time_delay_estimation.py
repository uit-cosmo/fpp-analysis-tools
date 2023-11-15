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


@dataclass
class CCFitOptions:
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

    def __str__(self):
        """Return a string representation of the CCFitOptions object."""
        return f"Fit Window: {self.fit_window}, Initial Guess: {self.initial_guess}, Interpolate: {self.interpolate}"


@dataclass
class CAOptions:
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

    def __str__(self):
        """Return a string representation of the CAOptions object."""
        return (
            f"Min Threshold: {self.min_threshold}, Max Threshold: {self.max_threshold}, Delta: {self.delta},"
            f" Window: {self.window}, Interpolate: {self.interpolate}, Verbose: {self.verbose}"
        )


@dataclass
class CCOptions:
    def __init__(
        self,
        cc_window: float = None,
        minimum_cc_value: float = 0.5,
        running_mean: bool = True,
        running_mean_window_max: int = 7,
        interpolate: bool = False,
    ):
        """
        - cc_window: int time lag window for the cross-correlation function computation.
        If set to T, the cross-correlation function is computed for time lags [-T, T]. If set to None,
        100 sampling times will be used as default
        - minimum_cc_value: float The cross-correlation maximum has to be at least this value, if less
        no estimate is performed.
        - running_mean: bool, if True, a running mean is applied to the estimated cross-correlation.
        The length of the running mean is determined as the minimum length such that the
        resulting cross-correlation has only one local maximum up to a factor 2. If the running mean
        window exceeds running_mean_window_max, a None result is returned and a warning is printed,
        indicating that either the ccf is nonunimodal, or that the data is too poor to estimate a time
        delay.
        - interpolate: If True the maximizing time lags are found by interpolation.
        """
        self.cc_window = cc_window
        self.minimum_cc_value = minimum_cc_value
        self.running_mean = running_mean
        self.window_max = running_mean_window_max
        self.interpolate = interpolate

    def __str__(self):
        """Return a string representation of the CCOptions object."""
        return (
            f"CC Window: {self.cc_window}, Minimum CC Value: {self.minimum_cc_value}, Running Mean: {self.running_mean},"
            f" Running Mean Window Max: {self.window_max}, Interpolate: {self.interpolate}"
        )


class TDEMethod(Enum):
    """Possible implemented methods.

    CC = cross-correlation based,
    CA = conditional average based,
    CCFit = cross-correlation fit based
    """

    CC = 1
    CA = 2
    CCFit = 3


class TDEDelegator:
    def __init__(self, method: TDEMethod, options, cache):
        self.method = method
        self.options = options
        self.cache = cache
        self.results = {}

    def estimate_time_delay(self, p1, p0, ds: utils.ImagingDataInterface):
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

    def estimate_time_delay_uncached(self, p1, p0, ds: utils.ImagingDataInterface):
        extra_debug_info = "between pixels {} and {}".format(p1, p0)
        x = ds.get_signal(p1[0], p1[1])
        y = ds.get_signal(p0[0], p0[1])
        dt = ds.get_dt()

        if ds.is_pixel_dead(p0[0], p0[1]) or ds.is_pixel_dead(p1[0], p1[1]):
            return None, None, None

        match self.method:
            case TDEMethod.CC:
                return estimate_time_delay_ccf(x, y, dt, self.options, extra_debug_info)
            case TDEMethod.CA:
                return estimate_time_delay_ccond_av_max(
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


def estimate_time_delay_ccf(
    x: np.ndarray,
    y: np.ndarray,
    dt: float,
    options: CCOptions,
    extra_debug_info: str = "",
):
    """
    Estimates the average time delay between to signals by finding the time
    lag that maximizes the cross-correlation function.
    Arguments:
        - x, y: Signals
        - dt: Sampling time
        - options: CCOptions, estimation options, see documentation for class CCOptions.
        - extra_debug_info: String to be appended to warnings printed by this function.
    Returns:
        td Estimated time delay
        ccf Cross-correlation value at the estimated time delay

    """
    ccf_times, ccf = cf.corr_fun(x, y, dt=dt, biased=True, norm=True)

    # Cut ccf to the window of interest.
    find_window = np.abs(ccf_times) < (
        options.cc_window if options.cc_window is not None else 100 * dt
    )
    ccf_times = ccf_times[find_window]
    ccf = ccf[find_window]

    max_index = np.argmax(ccf)
    max_time, ccf_value = ccf_times[max_index], ccf[max_index]
    if ccf_value < options.minimum_cc_value:
        warnings.warn(
            "CCF value too low to perform time delay estimation " + extra_debug_info
        )
        return None, None, None

    if not options.running_mean:
        if not options.interpolate:
            return max_time, ccf_value, 0

        max_time_interpolate = _find_maximum_interpolate(
            ccf_times, ccf, extra_debug_info
        )

        return max_time_interpolate, ccf_value, 0

    success, ccf, n = _run_mean_and_locate_maxima(
        ccf, max_run_window_size=options.window_max, extra_debug_info=extra_debug_info
    )
    if not success:
        return None, None, None
    if n > 1:
        ccf_times = ccf_times[int(n / 2) : -int(n / 2)]

    max_index = np.argmax(ccf)
    max_time, ccf_value = ccf_times[max_index], ccf[max_index]
    if not options.interpolate:
        return max_time, ccf_value, 0

    max_time_interpolate = _find_maximum_interpolate(ccf_times, ccf, extra_debug_info)

    return max_time_interpolate, ccf_value, 0


def get_time_delay_ccmax_rm_data(
    x: np.ndarray,
    y: np.ndarray,
    dt: float,
    options: CCOptions = CCOptions(),
):
    """Returns all data relevant for the method
    estimate_time_delay_ccmax_running_mean.

    if options.running_mean = False, and interpolate = False, return max_time, ccf_value,
    ccf_times, ccf

    if options.running_mean = False, and interpolate = True, return max_time_interpolate,
    ccf_value, ccf_times, spline

    if options.running_mean = True, and interpolate = False (or if interpolate = True, but
    running_mean does not succeed) return max_time, max_ccf_value, ccf_times_rm, ccf_rm

    if options.running_mean = False, and interpolate = False, return ccf_times, ccf, ccf_times_rm,
    ccf_rm, max_time_interpolate, max_ccf_value, spline

    Returns:
        ccf_times Times for ccf array without running mean
        ccf cross correlation function array without running mean
        ccf_times_rm Times for ccf array with running mean
        ccf_rm ccf array with running mean
        td Estimated time delay
        max_ccf_value Max cross-correlation value
    """
    ccf_times, ccf = cf.corr_fun(x, y, dt=dt, biased=True, norm=True)

    # Cut ccf to the window of interest.
    find_window = np.abs(ccf_times) < (
        options.cc_window if options.cc_window is not None else 100 * dt
    )
    ccf_times = ccf_times[find_window]
    ccf = ccf[find_window]

    if not options.running_mean:
        max_index = np.argmax(ccf)
        max_time, ccf_value = ccf_times[max_index], ccf[max_index]
        if not options.interpolate:
            return max_time, ccf_value, ccf_times, ccf

        max_time_interpolate, spline = _find_maximum_interpolate(
            ccf_times, ccf, return_spline=True
        )

        return max_time_interpolate, ccf_value, ccf_times, spline

    success, ccf_rm, n = _run_mean_and_locate_maxima(
        ccf, max_run_window_size=options.window_max
    )
    ccf_times_rm = ccf_times

    max_index = np.argmax(ccf_rm)
    max_time, max_ccf_value = ccf_times_rm[max_index], ccf_rm[max_index]
    if n > 1:
        ccf_times_rm = ccf_times[int(n / 2) : -int(n / 2)]
    if not success:
        return max_time, max_ccf_value, ccf_times_rm, ccf_rm

    if not options.interpolate:
        print(
            "You are getting four variables: max_time, max_ccf_value, ccf_times_rm, ccf_rm. Use them with care."
        )
        return max_time, max_ccf_value, ccf_times_rm, ccf_rm

    max_time_interpolate, spline = _find_maximum_interpolate(
        ccf_times, ccf, return_spline=True
    )

    return (
        ccf_times,
        ccf,
        ccf_times_rm,
        ccf_rm,
        max_time_interpolate,
        max_ccf_value,
        spline,
    )


def _run_mean_and_locate_maxima(ccf, max_run_window_size=7, extra_debug_info=""):
    ccf_mean = ccf
    n = 1
    while (lm := _count_local_maxima(ccf_mean)) != 1:
        if lm == 0:
            warnings.warn(
                "Cross-correlation function has no local maxima. " + extra_debug_info
            )
            return False, ccf_mean, n

        if n + 2 > max_run_window_size:
            warnings.warn("Maximum running window achieved " + extra_debug_info)
            return False, ccf_mean, n

        n = n + 2
        ccf_mean = np.convolve(ccf, np.ones(n) / n, mode="valid")
    return True, ccf_mean, n


def _count_local_maxima(ccf):
    local_maxima = np.array([False] * len(ccf))
    local_maxima[1:-1] = np.logical_and(ccf[1:-1] > ccf[2:], ccf[1:-1] > ccf[:-2])

    # Only count the local maxima that are at least half the value of the global maxima
    args = ccf[np.where(local_maxima)[0]]
    if len(args) == 0:
        return 0

    elegible_local_maxima = np.where(args > 0.5 * max(args))[0]
    return len(elegible_local_maxima)


def _find_maximum_interpolate(x, y, extra_debug_info, return_spline=False):
    from scipy.interpolate import InterpolatedUnivariateSpline

    # Taking the derivative and finding the roots only work if the spline degree is at least 4.
    spline = InterpolatedUnivariateSpline(x, y, k=4)
    possible_maxima = spline.derivative().roots()
    possible_maxima = np.append(
        possible_maxima, (x[0], x[-1])
    )  # also check the endpoints of the interval
    values = spline(possible_maxima)

    max_index = np.argmax(values)
    max_time = possible_maxima[max_index]
    if max_time == x[0] or max_time == x[-1]:
        warnings.warn(
            "Maximization on interpolation yielded a maximum in the boundary!"
            + extra_debug_info
        )

    if return_spline:
        return max_time, spline
    return max_time


def estimate_time_delay_ccond_av_max(
    x: np.ndarray,
    y: np.ndarray,
    dt: float,
    cond_av_eo: CAOptions,
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
