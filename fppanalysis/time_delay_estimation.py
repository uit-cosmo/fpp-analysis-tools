from scipy.stats import gamma
import correlation_function as cf


def estimate_delays(x, y, dt, distribution=gamma, ax=None, plot="full", min_cutoff=0):
    """
    Use:
       estimate_delays(x, y, dt, distribution=gamma, ax=None, plot="full", min_cutoff=0)

    Estimates the time delay distribution parameters for the pulses propagating from two measurement points.
    Time series x, y, corresponding to each point measurement must be provided. The time delay distribution
    is assumed to follow a family distribution given by the argument distribution. Plots relevant autocorrelation
    and cross-correlation functions if a matplotlib ax is provided.

    Wrapper function for run_mean(), computes running mean and rms of S.
    To compute the running standard deviation of S, the running mean is
    subtracted from the signal.
    The running rms divides by window, not (window-1).

    Input:
        x: Time series ...................... (N,) np array
        x: Time series ...................... (N,) np array
        distribution: Assumed distribution .... class implementing scipy.stats.rv_continuous
        ax: Optional, if a matplotlib.pyplot.axis is provided, relevant plots will be plotted. These are meant to help
        understand the underlying principles of the optimization, the plots are not suitable for scientific publication
        .
        plot: String, if "full", plots more stuff.
        min_cutoff: I don't remember.
    Output:
        avg: Average delay time.
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
        pdf = get_pdf(params, tauR)
        res = fftconvolve(autocorrelation, pdf, "same") * dt
        res /= max(res)
        return np.sum((res - R) ** 2)

    def get_params():
        if distribution in [uniform, norm]:
            return [1, 1]
        return np.ones(1 + distribution.numargs)

    # We cut the cross correlation in the middle half to avoid noise near the ends

    parameters = get_params()

    tauR, R = cf.corr_fun(x, y, dt=dt, biased=False)
    _, autocorrelation = cf.corr_fun(x, x, dt=dt, biased=False)

    R = R[np.abs(tauR) < max(tauR) / 2]
    autocorrelation = autocorrelation[np.abs(tauR) < max(tauR) / 2]
    tauR = tauR[np.abs(tauR) < max(tauR) / 2]

    max_cross_corr = max(tauR[np.argmax(R)], min_cutoff)
    parameters[0] = max_cross_corr
    domain_cutoff = max_cross_corr * 100

    domain = np.abs(tauR) < domain_cutoff
    tauR = tauR[domain]
    R = R[domain]
    autocorrelation = autocorrelation[domain]
    R = R / max(R)

    minimization = minimize(
        gamma_error, parameters, method="Nelder-Mead", options={"maxiter": 10000}
    )
    if not minimization.success:
        print("Optimization failed!!!")
    arg = minimization.x[:-1]
    scale = minimization.x[0]
    avg = get_average(minimization.x)

    if ax is not None:
        ax.plot(
            tauR, R, label=r"$\wh{R_{\tilde{\Phi}, \tilde{\Psi}}}(r)$", color="blue"
        )
        convo = fftconvolve(autocorrelation, get_pdf(minimization.x, tauR), "same")
        convo /= max(convo)
        ax.plot(
            tauR,
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
            axin.plot(tauR, get_pdf(minimization.x, tauR))
            axin.set_xlim(-max_cross_corr, 5 * max_cross_corr)
            axin.set_title("Params: {}".format(minimization.x), fontsize=8)

    return avg
