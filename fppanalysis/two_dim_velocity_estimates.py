import fppanalysis.time_delay_estimation as tde

import numpy as np
import xarray as xr


def get_2d_velocities_from_time_delays(delta_tx, delta_ty, delta_x, delta_y):
    """
    Estimates radial and poloidal velocities given the input parameters:
    Input:
         delta_tx Estimation of the time delay between radially separated points.
         delta_ty Estimation of the time delay between poloidally separated points.
         delta_x Spatial separation between radially separated points.
         delta_y Spatial separation between poloidally separated points.

    These quantities should be obtained from three pixel points: a reference pixel point,
    a pixel point separated radially, and a pixel point separated poloidally.
    Returns:
         vx Radial velocity
         vy Poloidal velocity
    """
    if delta_tx == 0:
        return 0, delta_y / delta_ty
    if delta_ty == 0:
        return delta_x / delta_tx, 0
    fx = delta_x / delta_tx
    fy = delta_y / delta_ty
    return fx / (1 + (fx / fy) ** 2), fy / (1 + (fy / fx) ** 2)


def _get_rz(x, y, ds):
    # Sajidah's format
    if hasattr(ds, "time"):
        return ds.R.isel(x=x, y=y).values, ds.Z.isel(x=x, y=y).values
    # 2d code
    if hasattr(ds, "t"):
        return ds.x.isel(x=x).values, ds.y.isel(y=y).values
    raise "Unknown format"


def _get_rz_full(ds):
    # Sajidah's format
    if hasattr(ds, "time"):
        shape = (len(ds.x.values), len(ds.y.values))
        R = np.zeros(shape=shape)
        Z = np.zeros(shape=shape)
        for x in ds.x.values:
            for y in ds.y.values:
                R[x, y] = ds.R.isel(x=x, y=y).values
                Z[x, y] = ds.Z.isel(x=x, y=y).values
        return R, Z
    # 2d code
    if hasattr(ds, "t"):
        return np.meshgrid(ds.x.values, ds.y.values)
    raise "Unknown format"


def _get_signal(x, y, ds):
    # Sajidah's format
    if hasattr(ds, "time"):
        return ds.isel(x=x, y=y).dropna(dim="time", how="any")["frames"].values
    # 2d code
    if hasattr(ds, "t"):
        return ds.isel(x=x, y=y).dropna(dim="t", how="any")["n"].values
    raise "Unknown format"


def _get_dt(ds):
    # Sajidah's format
    if hasattr(ds, "time"):
        times = ds["time"]
        return times[1].values - times[0].values
    # 2d code
    if hasattr(ds, "t"):
        times = ds["t"]
        return times[1].values - times[0].values
    raise "Unknown format"


def _get_time(x, y, ds):
    # Sajidah's format
    if hasattr(ds, "time"):
        return ds.isel(x=x, y=y).time.values


def _estimate_time_delay(
    x: np.ndarray,
    x_t: np.ndarray,
    y: np.ndarray,
    y_t: np.ndarray,
    method: str,
    dt: float,
    cut_off_freq=1e3,
    threshold=2.5,
):
    """Estimates the average time delay between to signals by finding the time
    lag that maximizies either the cross-correlation function or the cross
    conditional average of signal x when signal y is larger than threshold.
    Either one of these method must be specified in method-argument.

    Input:
        x: Signal to be conditionally averaged
        y: Reference signal
        x_t: Time of signal x
        y_t: Time of signal y
        method: 'cross_corr' or 'cond_av' for either cross correlation or cross conditional average
        if method == 'cross_corr':
            dt: Time step
        if method == 'cond_av':
            cut_off_freq: Cut off frequency to decide window size for running moments
            threshold: Threshold value for conditional average. Defualt value is set to 2.5.

    Returns:
        if method == 'cross_corr':
            td Estimated time delay
            C Cross correlation at a time lag td.

        if method == 'cond_av':
            td: Estimated time delay
            C: Cross conditional variance at a time lag td.
            events: Number of events larger than 2.5 the mean value
    """

    if method == "cross_corr":
        delta_t, c = tde.estimate_time_delay_ccmax(x=x, y=y, dt=dt)
        return delta_t, c

    if method == "cond_av":
        delta_t, cond_variance, events = tde.estimate_time_delay_ccond_av_max(
            x=x, x_t=x_t, y=y, y_t=y_t
        )

        # Confidence when velocities are estimated from
        # cross conditional averge is: 1 - conditional variance
        c = 1 - cond_variance

        return delta_t, c

    else:
        raise Exception("Method must be either cross_corr or cond_av")


def _estimate_velocities_given_points(
    p0, p1, p2, ds, method: str, cut_off_freq=1e3, threshold=2.5
):
    """Estimates radial and poloidal velocity from estimated time delay
    either from cross conditional average (if cross_cond_av = True)
    between the pixels or cross correlation (if cross_cond_av = False).
    In cross conditional average, time array is required in order to normalize
    the signal before averageing.
    """
    dt = _get_dt(ds)
    r0, z0 = _get_rz(p0[0], p0[1], ds)
    r1, z1 = _get_rz(p1[0], p1[1], ds)
    r2, z2 = _get_rz(p2[0], p2[1], ds)
    signal0 = _get_signal(p0[0], p0[1], ds)
    signal1 = _get_signal(p1[0], p1[1], ds)
    signal2 = _get_signal(p2[0], p2[1], ds)
    time0 = _get_time(p0[0], p0[1], ds)
    time1 = _get_time(p1[0], p1[1], ds)
    time2 = _get_time(p2[0], p2[1], ds)

    if len(signal0) == 0 or len(signal1) == 0 or len(signal2) == 0:
        return None

    delta_ty, cy = _estimate_time_delay(
        x=signal2,
        x_t=time2,
        y=signal0,
        y_t=time0,
        method=method,
        dt=dt,
        cut_off_freq=1e3,
        threshold=2.5,
    )
    delta_tx, cx = _estimate_time_delay(
        x=signal1,
        x_t=time1,
        y=signal0,
        y_t=time0,
        method=method,
        dt=dt,
        cut_off_freq=1e3,
        threshold=2.5,
    )

    confidence = min(cx, cy)

    return (
        *get_2d_velocities_from_time_delays(delta_tx, delta_ty, r1 - r0, z2 - z0),
        confidence,
    )


def _is_within_boundaries(p, ds):
    return 0 <= p[0] < ds.sizes["x"] and 0 <= p[1] < ds.sizes["y"]


def estimate_velocities_for_pixel(
    x, y, ds: xr.Dataset, method: str, cut_off_freq=1e3, threshold=2.5
):
    """Estimates radial and poloidal velocity for a pixel with indexes x,y
    using all four possible combinations of nearest neighbour pixels (x-1, y),
    (x, y+1), (x+1, y) and (x, y-1). Dead-pixels (stored as np.nan arrays) are
    ignored. Pixels outside the coordinate domain are ignored. Time delay
    estimation is performed by maximizing the cross- correlation function. The
    confidence of the estimation is a value in the interval (0, 1) given by the
    mean of the confidences for each combination, which is given by the minimum
    of the maximums of the two cross-correlations involved (good luck
    understanding this last sentence :))

    Input:
        x: pixel index x
        y: pixel index y
        ds: xarray Dataset
        method: 'cross_corr' or 'cond_av'
        if method == 'cond_av':
            cut_off_freq: Cut off frequency to decide window size for running moments
            threshold: Threshold value for conditional average. Defualt value is set to 2.5.

    Returns:
        vx Radial velocity
        vy Poloidal velocity
        c Confidence on the estimation
    """

    h_neighbors = [(x - 1, y), (x + 1, y)]
    v_neighbors = [(x, y - 1), (x, y + 1)]
    results = [
        _estimate_velocities_given_points((x, y), px, py, ds, method)
        for px in h_neighbors
        if _is_within_boundaries(px, ds)
        for py in v_neighbors
        if _is_within_boundaries(py, ds)
    ]
    results = [r for r in results if r is not None]
    if len(results) == 0:  # If (x,y) is dead we cannot estimate
        return None, None, None
    mean_vx = sum(map(lambda r: r[0], results)) / len(results)
    mean_vy = sum(map(lambda r: r[1], results)) / len(results)
    confidence = sum(map(lambda r: r[2], results)) / len(results)
    return mean_vx, mean_vy, confidence


def estimate_velocity_field(
    ds: xr.Dataset, method: str, cut_off_freq=1e3, threshold=2.5
):
    """
    Given a dataset ds with GPI data in a format produced by https://github.com/sajidah-ahmed/cmod_functions,
    computed the velocity field. The estimation takes into account poloidal flows as described in
    the 2D filament model. For each pixel, the velocities are estimated using the given pixel, and two neighbour
    pixels: the right neighbour and the down neighbour. The return objects are matrices of the size of the
    GPI grid, from which the velocity field can be easily plotted via f.e matplotlib.quiver.

    Input:
        ds: xarray Dataset
        method: 'cross_corr' or 'cond_av'
        if method == 'cond_av':
            cut_off_freq: Cut off frequency to decide window size for running moments
            threshold: Threshold value for conditional average. Defualt value is set to 2.5.

    Returns:
        vx Radial velocities
        vy Poloidal velocities
        confidences Maximum value of the cross-correlations at each pixel.
        R Radial positions
        Z Radial positions
    """
    shape = (len(ds.x.values), len(ds.y.values))
    vx = np.zeros(shape=shape)
    vy = np.zeros(shape=shape)
    confidences = np.zeros(shape=shape)
    R, Z = _get_rz_full(ds)

    for i in range(0, shape[0]):
        for j in range(0, shape[1]):
            try:
                vx[i, j], vy[i, j], confidences[i, j] = estimate_velocities_for_pixel(
                    i,
                    j,
                    ds,
                    method,
                    cut_off_freq=1e3,
                    threshold=2.5,
                )
            except:
                print(
                    "Issues estimating velocity for pixel",
                    i,
                    j,
                    "Run estimate_velocities_for_pixel(i, j, ds) to get a detailed error stacktrace",
                )

    vx[np.isnan(vx) | np.isinf(vx)] = 0
    vy[np.isnan(vy) | np.isinf(vy)] = 0
    return vx, vy, confidences, R, Z
