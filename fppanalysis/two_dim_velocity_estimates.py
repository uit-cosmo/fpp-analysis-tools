import warnings

import fppanalysis.time_delay_estimation as tde
import numpy as np
import xarray as xr
from dataclasses import dataclass


class EstimationOptions:
    def __init__(
        self,
        method: str = "cross_corr",
        use_2d_estimation: bool = True,
        neighbors_ccf_min_lag: int = 0,
        interpolate: bool = True,
        num_cores: int = 1,
        min_threshold: float = 2.5,
        max_threshold: float = np.inf,
        delta: float = None,
        window: bool = False,
        verbose: bool = False,
        ccf_fit_eo: tde.CcfFitEstimationOptions = tde.CcfFitEstimationOptions()
    ):
        """
        Estimation options for velocity estimation method.

        - method: 'cross_corr' or 'cond_av'
        - use_2d_estimation: [bool] If False, use 1 dimensional method to estimate velocities.
        - neighbors_ccf_min_lag: Integer, checks that the maximal correlation between adjacent
        pixels occurs at a time smaller than neighbors_ccf_min_lag multiples of the discretization
        time. If that's not the case, the next neighbor will be used, and so on until a
        neighbor pixel is found complient to this condition. If set to -1, no condition will
        be applied.
        - interpolate: If True the maximizing time lags are found by interpolation.
        - num_cores: Number of cores to use.
        - min_threshold: Used only if method = "cond_av". min threshold for conditional averaged events
        - max_threshold: Used only if method = "cond_av". max threshold for conditional averaged events
        - delta: Used only if method = "cond_av". If window = True, delta is the minimal distance between two peaks.
        - window: Used only if method = "cond_av". [bool] If True, delta also gives the minimal distance between peaks.
        - ccf_fit_eo: Time delay estimation options to be used if method = "cross_corr_fit"
        """
        self.method = method
        self.use_2d_estimation = use_2d_estimation
        self.neighbors_ccf_min_lag = neighbors_ccf_min_lag
        self.interpolate = interpolate
        self.num_cores = num_cores
        self.min_threshold = min_threshold
        self.max_threshold = max_threshold
        self.delta = delta
        self.window = window
        self.verbose = verbose
        self.ccf_fit_eo = ccf_fit_eo


@dataclass
class PixelData:
    """Data class containing estimated data from a single pixel.

    vx: Radial velocities
    vy: Poloidal velocities
    confidences:
        if method='cross_corr':
            Maximum value of the cross-correlations at each pixel.
        if method='cond_av':
            Conditional variance value at maximum cross conditional average for each pixel.
    R: Radial positions
    Z: Poloidal positions
    """

    r_pos: float = 0
    z_pos: float = 0
    vx: float = 0
    vy: float = 0
    confidence: float = 0
    events: int = 0


class MovieData:
    """Class containing estimated data for all pixels in a set. Return object
    from estimate_velocity_field function, the indexing format of MovieData
    matches that of the dataset input of estimate_velocity_field.

    Use getters to retrieve:
        vx: Radial velocities
        vy: Poloidal velocities
        confidences:
            if method='cross_corr':
                Maximum value of the cross-correlations at each pixel.
            if method='cond_av':
                Conditional variance value at maximum cross conditional average for each pixel.
        R: Radial positions
        Z: Poloidal positions

    Dead pixels have empty PixelData (null vx and vy).
    """

    def _set_pixel(self, items):
        i, j = items[0], items[1]
        try:
            return estimate_velocities_for_pixel(i, j, self.ds, self.estimation_options)
        except:
            print(
                "Issues estimating velocity for pixel",
                i,
                j,
                "Run estimate_velocities_for_pixel(i, j, ds, method, **kwargs) to get a detailed error stacktrace",
            )
        return PixelData()

    def __init__(self, ds, estimation_options: EstimationOptions):
        range_r, range_z = range(0, len(ds.x.values)), range(0, len(ds.y.values))
        self.r_dim = len(range_r)
        self.z_dim = len(range_z)
        self.ds = ds
        self.estimation_options = estimation_options
        self.pixels = [[PixelData() for _ in range_z] for _ in range_r]

        from pathos.multiprocessing import ProcessPool

        pool = ProcessPool(estimation_options.num_cores)
        results = pool.map(self._set_pixel, [(i, j) for i in range_r for j in range_z])
        for i in range_r:
            for j in range_z:
                self.pixels[i][j] = results[len(range_r) * j + i]

    def _get_field(self, field_name):
        return np.array(
            [[getattr(p, field_name) for p in pixel_row] for pixel_row in self.pixels]
        )

    def get_vx(self):
        return self._get_field("vx")

    def get_vy(self):
        return self._get_field("vy")

    def get_R(self):
        return self._get_field("r_pos")

    def get_Z(self):
        return self._get_field("z_pos")

    def get_events(self):
        return self._get_field("events")

    def get_confidences(self):
        return self._get_field("confidence")


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


def get_1d_velocities_from_time_delays(delta_tx, delta_ty, delta_x, delta_y):
    """
    Estimates radial and poloidal velocities from naive method
    given the input parameters:
    Input:
         delta_tx Estimation of the time delay between radially separated points.
         delta_ty Estimation of the time delay between poloidally separated points.
         delta_x Spatial separation between radially separated points.
         delta_y Spatial separation between poloidally separated points.

    These quantities should be obtained from two pixel points:
        radial direction: a reference pixel point and a pixel point separated radially
        poloidal direction: a reference pixel point and a pixel point separated poloidally.
    Returns:
         vx Radial velocity
         vy Poloidal velocity
    """
    vx = 0 if delta_tx == 0 else delta_x / delta_tx
    vy = 0 if delta_ty == 0 else delta_y / delta_ty

    return vx, vy


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
    # 2d code
    if hasattr(ds, "t"):
        return ds.t.values
    raise "Unknown format"


def _estimate_time_delay(
    x: np.ndarray,
    x_t: np.ndarray,
    y: np.ndarray,
    dt: float,
    estimation_options: EstimationOptions,
):
    match estimation_options.method:
        case "cross_corr":
            (delta_t, c), events = (
                tde.estimate_time_delay_ccmax(
                    x=x, y=y, dt=dt, interpolate=estimation_options.interpolate
                ),
                0,
            )
        case "cond_av":
            delta_t, c, events = tde.estimate_time_delay_ccond_av_max(
                x=x,
                x_t=x_t,
                y=y,
                min_threshold=estimation_options.min_threshold,
                max_threshold=estimation_options.max_threshold,
                delta=estimation_options.delta,
                window=estimation_options.window,
                interpolate=estimation_options.interpolate,
            )
        case "cross_corr_fit":
            (delta_t, c), events = (
                tde.estimate_time_delay_ccf_fit(
                    x=x, y=y, dt=dt, estimation_options=estimation_options.ccf_fit_eo
                ),
                0,
            )
        case "cross_corr_running_mean":
            (delta_t, c), events = (
                tde.estimate_time_delay_ccmax_running_mean(
                    x=x, y=y, dt=dt, interpolate=estimation_options.interpolate
                ),
                0,
            )
        case _:
            raise Exception("Method must be either cross_corr or cond_av")
    return delta_t, c, events


def _estimate_velocities_given_points(
    p0, p1, p2, ds, estimation_options: EstimationOptions
):
    """Estimates radial and poloidal velocity from estimated time delay either
    from cross conditional average between the pixels or cross correlation.

    This is specified in method argument.
    """
    dt = _get_dt(ds)
    r0, z0 = _get_rz(p0[0], p0[1], ds)
    r1, z1 = _get_rz(p1[0], p1[1], ds)
    r2, z2 = _get_rz(p2[0], p2[1], ds)
    signal0 = _get_signal(p0[0], p0[1], ds)
    signal1 = _get_signal(p1[0], p1[1], ds)
    signal2 = _get_signal(p2[0], p2[1], ds)
    time1 = _get_time(p1[0], p1[1], ds)
    time2 = _get_time(p2[0], p2[1], ds)

    if len(signal0) == 0 or len(signal1) == 0 or len(signal2) == 0:
        return None

    delta_ty, cy, events_y = _estimate_time_delay(
        x=signal2, x_t=time2, y=signal0, dt=dt, estimation_options=estimation_options
    )
    delta_tx, cx, events_x = _estimate_time_delay(
        x=signal1, x_t=time1, y=signal0, dt=dt, estimation_options=estimation_options
    )

    # If for some reason the time delay cannot be estimated, we return None
    if delta_tx is None or delta_ty is None:
        return None

    confidence = min(cx, cy)
    events = min(events_x, events_y)

    if estimation_options.use_2d_estimation:
        return (
            *get_2d_velocities_from_time_delays(delta_tx, delta_ty, r1 - r0, z2 - z0),
            confidence,
            events,
        )
    else:
        return (
            *get_1d_velocities_from_time_delays(delta_tx, delta_ty, r1 - r0, z2 - z0),
            confidence,
            events,
        )


def _is_within_boundaries(p, ds):
    return 0 <= p[0] < ds.sizes["x"] and 0 <= p[1] < ds.sizes["y"]


def _check_ccf_constrains(p0, p1, ds, neighbors_ccf_min_lag: int):
    """Returns true if the time lag that maximizes the cross-correlation
    function measure at p0 and p1 is not zero
    """
    import fppanalysis.correlation_function as cf

    signal0 = _get_signal(p0[0], p0[1], ds)
    signal1 = _get_signal(p1[0], p1[1], ds)

    if len(signal1) == 0:
        warnings.warn(
            "Pixel {} is dead and cannot be used as a neighbor pixel of {}. Updating.".format(
                p1, p0
            )
        )
        return False

    ccf_times, ccf = cf.corr_fun(
        signal0, signal1, dt=_get_dt(ds), biased=True, norm=True
    )
    ccf = ccf[np.abs(ccf_times) < max(ccf_times) / 2]
    ccf_times = ccf_times[np.abs(ccf_times) < max(ccf_times) / 2]
    max_index = np.argmax(ccf)

    fulfills_constrain = np.abs(
        ccf_times[max_index]
    ) >= neighbors_ccf_min_lag * _get_dt(ds)
    if not fulfills_constrain:
        warnings.warn(
            "Pixel {} does not fulfill cross-correlation time lag condition with respect to pixel of {}."
            " Updating.".format(p1, p0)
        )

    return fulfills_constrain


def _find_neighbors(x, y, ds: xr.Dataset, neighbors_ccf_min_lag: int):
    def should_hopp_pixel(p):
        # if neighbors_ccf_min_lag is set to -1, we don't hopp (see docs).
        if neighbors_ccf_min_lag == -1:
            return False
        return _is_within_boundaries(p, ds) and not _check_ccf_constrains(
            (x, y), p, ds, neighbors_ccf_min_lag
        )

    left = -1
    while should_hopp_pixel((x + left, y)):
        left -= 1

    right = 1
    while should_hopp_pixel((x + right, y)):
        right += 1

    up = 1
    while should_hopp_pixel((x, y + up)):
        up += 1

    down = -1
    while should_hopp_pixel((x, y + down)):
        down -= 1

    return [(x + left, y), (x + right, y)], [(x, y + down), (x, y + up)]


def estimate_velocities_for_pixel(
    x, y, ds: xr.Dataset, estimation_options: EstimationOptions = EstimationOptions()
):
    """Estimates radial and poloidal velocity for a pixel with indexes x,y
    using all four possible combinations of nearest neighbour pixels (x-1, y),
    (x, y+1), (x+1, y) and (x, y-1). Dead-pixels (stored as np.nan arrays) are
    ignored. Pixels outside the coordinate domain are ignored. Time delay
    estimation is performed by maximizing either the cross- correlation
    function or cross conditional average function, which is specified in input
    argument 'method'.

    If time delay estimation is performed by maximizing the cross correlation function,
    the confidence of the estimation is a value in the interval (0, 1) given by the
    mean of the confidences for each combination, which is given by the minimum
    of the maximums of the two cross-correlations involved.

    If time delay estimation is performed by maximizing the cross conditional average function,
    the confidence of the estimation is a value in the interval (0, 1) given by the
    cross conditional variance for each event. OBS: We return 1-CV for cross conditional variance.

    Input:
        x: pixel index x
        y: pixel index y
        ds: xarray Dataset
        estimation_options: EstimationOptions class including all estimation parameters, if not set
        the method will be based on cross-correlation function.


    Returns:
        PixelData: Object containing radial and poloidal velocities and method-specific data.
    """
    r_pos, z_pos = _get_rz(x, y, ds)

    # If the reference pixel is dead, return empty data right away
    if len(_get_signal(x, y, ds)) == 0:
        return PixelData(r_pos=r_pos, z_pos=z_pos)

    h_neighbors, v_neighbors = _find_neighbors(
        x, y, ds, estimation_options.neighbors_ccf_min_lag
    )
    results = [
        _estimate_velocities_given_points((x, y), px, py, ds, estimation_options)
        for px in h_neighbors
        if _is_within_boundaries(px, ds)
        for py in v_neighbors
        if _is_within_boundaries(py, ds)
    ]

    results = [r for r in results if r is not None]
    if len(results) == 0:  # If no neighbor pixels are found we cannot estimate
        return PixelData(r_pos=r_pos, z_pos=z_pos)

    mean_vx = sum(map(lambda r: r[0], results)) / len(results)
    mean_vy = sum(map(lambda r: r[1], results)) / len(results)
    confidence = sum(map(lambda r: r[2], results)) / len(results)
    events = sum(map(lambda r: r[3], results)) / len(results)

    return PixelData(
        r_pos=r_pos,
        z_pos=z_pos,
        vx=mean_vx,
        vy=mean_vy,
        confidence=confidence,
        events=events,
    )


def estimate_velocity_field(
    ds: xr.Dataset, estimation_options: EstimationOptions = EstimationOptions()
) -> MovieData:
    """Computes the velocity field of a given dataset ds with GPI data in a
    format produced by https://github.com/sajidah-ahmed/cmod_functions. The
    estimation takes into account poloidal flows as described in the 2D
    filament model. For each pixel, the velocities are estimated using the
    given pixel, and two neighbour pixels: the right neighbour and the down
    neighbour. The velocities are estimated from a time delay estimation
    performed by maximizing either the cross- correlation function or cross
    conditional average function, which is specified in input argument
    'method'.

    If time delay estimation is performed by maximizing the cross correlation function,
    the confidence of the estimation is a value in the interval (0, 1) given by the
    mean of the confidences for each combination, which is given by the minimum
    of the maximums of the two cross-correlations involved.

    If time delay estimation is performed by maximizing the cross conditional average function,
    the confidence of the estimation is a value in the interval (0, 1) given by the
    cross conditional variance for each event. OBS: We return 1-CV for cross conditional variance.

    The return objects are matrices of the size of the GPI grid,
    from which the velocity field can be easily plotted via f.e matplotlib.quiver.

    Input:
        ds: xarray Dataset
        estimation_options: EstimationOptions class including all estimation parameters, if not set
        the method will be based on cross-correlation function.


    Returns:
        movie_data: Class containing estimation data about all pixels
    """

    return MovieData(ds, estimation_options)
