from scipy.stats import gamma
import correlation_function as cf


def estimate_delays(x, y, dt, distribution=gamma, ax=None, plot="full", min_cutoff=0):
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
        x: Time series ...................... (N,) np array
        distribution: Assumed distribution .... class implementing scipy.stats.rv_continuous
        ax: Optional, if a matplotlib.pyplot.axis is provided, relevant plots will be plotted. These are meant to help
        understand the underlying principles of the optimization, the plots are not suitable for scientific publication
        .
        plot: String, if "full", plots more stuff.
        min_cutoff: An upper bound for the cross-correlation maxima. Usage: If working with short time series
        or noisy data, it can be helpful for the method to set this value to an upper bound for the time that
        maximizes the cross-correlation, that is, a time such that you are sure that the cross-correlation is
        maximized before it.
    Output:
        avg: Average delay time
        arg: Possible distribution shape parameter. None if distribution is shapeless
        scale: Distribution scale parameter.
    """
    import numpy as np
    from scipy.stats import uniform, norm
    from scipy.signal import fftconvolve
    from scipy.optimize import minimize

    def get_pdf(params, tauR):
        if distribution == uniform:
            pdf = np.zeros(len(tauR))
            shape = 1 / (1 + params[1] ** 2)
            low = params[0] * (1 - shape)
            high = params[0] * (1 + shape)
            pdf[np.logical_and(tauR < high, tauR >= low)] = 1 / (high - low)
            return pdf
        if distribution == norm:
            return distribution.pdf(tauR, loc=params[0], scale=params[1])
        if distribution.numargs > 0:
            pdf = distribution.pdf(tauR, params[1], loc=0, scale=params[0])
            if distribution == gamma and params[1] < 1:
                pdf[tauR == 0] = 0
            return pdf
        else:
            return distribution.pdf(tauR, loc=0, scale=params[0])

    def get_average(params):
        if distribution == uniform:
            return params[0]
        if distribution == norm:
            return params[0]
        if distribution.numargs > 0:
            return distribution.stats(params[1], scale=scale, loc=0, moments="m")
        else:
            return distribution.stats(scale=scale, loc=0, moments="m")

    def gamma_error(params):
        pdf = get_pdf(params, ccf_times)
        res = fftconvolve(est_acf, pdf, "same") * dt
        res /= max(res)
        return np.sum((res - est_ccf) ** 2)

    def get_params():
        if distribution in [uniform, norm]:
            return [1, 1]
        return np.ones(1 + distribution.numargs)

    parameters = get_params()

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

    minimization = minimize(
        gamma_error, parameters, method="Nelder-Mead", options={"maxiter": 10000}
    )
    if not minimization.success:
        print("Optimization failed!!!")

    num_args = len(minimization.x)
    arg = minimization.x[:-1] if num_args > 1 else None
    scale = minimization.x[0]
    avg = get_average(minimization.x)

    if ax is not None:
        ax.plot(
            ccf_times,
            est_ccf,
            label=r"$\wh{R_{\tilde{\Phi}, \tilde{\Psi}}}(r)$",
            color="blue",
        )
        convo = fftconvolve(est_acf, get_pdf(minimization.x, ccf_times), "same")
        convo /= max(convo)
        ax.plot(
            ccf_times,
            convo,
            label=r"$\ave{\wh{\rho_\phi} \left( \frac{ r-d } {\tau} \right)}_d$",
            color="red",
        )

        ax.legend()
        ax.grid(True)
        ax.set_xlim(-10 * max_cross_corr, 20 * max_cross_corr)
        ax.set_ylim(0, 1.2)

        ax.set_xlabel(r"$r$")

        if plot == "full":
            ax.set_title(
                "Distribution: {} Error {:.2g}".format(
                    distribution.name, minimization.fun
                )
            )
            from mpl_toolkits.axes_grid1.inset_locator import inset_axes

            axin = inset_axes(
                ax,
                width="20%",  # width = 30% of parent_bbox
                height=0.8,  # height : 1 inch
                loc=4,
            )
            axin.plot(ccf_times, get_pdf(minimization.x, ccf_times))
            axin.set_xlim(-max_cross_corr, 5 * max_cross_corr)
            axin.set_title("Params: {}".format(minimization.x), fontsize=8)

    return avg, arg, scale
