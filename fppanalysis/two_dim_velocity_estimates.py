import fppanalysis.time_delay_estimation as tde
import numpy as np
import xarray as xr
from dataclasses import dataclass


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

    def __init__(self, range_r, range_z, func):
        self.r_dim = len(range_r)
        self.z_dim = len(range_z)
        self.pixels = [[PixelData() for i in range_z] for j in range_r]

        for i in range_r:
            for j in range_z:
                try:
                    self.pixels[i][j] = func(i, j)
                except:
                    print(
                        "Issues estimating velocity for pixel",
                        i,
                        j,
                        "Run estimate_velocities_for_pixel(i, j, ds, method, **kwargs) to get a detailed error stacktrace",
                    )

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
    if delta_tx == 0:
        return 0, delta_y / delta_ty
    if delta_ty == 0:
        return delta_x / delta_tx, 0
    fx = delta_x / delta_tx
    fy = delta_y / delta_ty
    return fx, fy 

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
    method: str,
    dt: float,
    **kwargs: dict
):

    if method == "cross_corr":
        (delta_t, c), events = tde.estimate_time_delay_ccmax(x=x, y=y, dt=dt), 0

    elif method == "cond_av":
        delta_t, c, events = tde.estimate_time_delay_ccond_av_max(
            x=x,
            x_t=x_t,
            y=y,
            min_threshold=kwargs["min_threshold"],
            max_threshold=kwargs["max_threshold"],
            delta=kwargs["delta"],
            window=kwargs["window"],
        )

    else:
        raise Exception("Method must be either cross_corr or cond_av")

    return delta_t, c, events


def _estimate_velocities_given_points(p0, p1, p2, ds, method: str, naive: bool, **kwargs: dict):
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
        x=signal2,
        x_t=time2,
        y=signal0,
        method=method,
        dt=dt,
        **kwargs,
    )
    delta_tx, cx, events_x = _estimate_time_delay(
        x=signal1,
        x_t=time1,
        y=signal0,
        method=method,
        dt=dt,
        **kwargs,
    )

    confidence = min(cx, cy)
    events = min(events_x, events_y)

    velocities_1d = get_1d_velocities_from_time_delays(delta_tx, delta_ty, r1 - r0, z2 - z0)
    velocities_2d = get_2d_velocities_from_time_delays(delta_tx, delta_ty, r1 - r0, z2 - z0)
    
    return (
        velocities_2d if not naive else velocities_1d,
        confidence,
        events,
    )


def _is_within_boundaries(p, ds):
    return 0 <= p[0] < ds.sizes["x"] and 0 <= p[1] < ds.sizes["y"]


def estimate_velocities_for_pixel(
    x, y, ds: xr.Dataset, method: str = "cross_corr", naive: bool = False, **kwargs: dict
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
        method: 'cross_corr' or 'cond_av'
        naive: [bool] If True, use 1D method to estimate velocities. If False, use 2D method.
        kwargs: kwargs used in 'cond_av'
            - min_threshold: min threshold for conditional averaged events
            - max_threshold: max threshold for conditional averaged events
            - delta: If window = True, delta is the minimal distance between two peaks.
            - window: [bool] If True, delta also gives the minimal distance between peaks.

    Returns:
        PixelData: Object containing radial and poloidal velocities and method-specific data.
    """

    h_neighbors = [(x - 1, y), (x + 1, y)]
    v_neighbors = [(x, y - 1), (x, y + 1)]
    results = [
        _estimate_velocities_given_points((x, y), px, py, ds, method, naive, **kwargs)
        for px in h_neighbors
        if _is_within_boundaries(px, ds)
        for py in v_neighbors
        if _is_within_boundaries(py, ds)
    ]
    results = [r for r in results if r is not None]
    r_pos, z_pos = _get_rz(x, y, ds)
    if len(results) == 0:  # If (x,y) is dead we cannot estimate
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
    ds: xr.Dataset, method: str = "cross_corr", naive: bool = False, **kwargs: dict
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
        method: 'cross_corr' or 'cond_av'
        naive: [bool] If True, use 1D method to estimate velocities. If False, use 2D method.
        kwargs: kwargs used in 'cond_av'
            - min_threshold: min threshold for conditional averaged events
            - max_threshold: max threshold for conditional averaged events
            - delta: If window = True, delta is the minimal distance between two peaks.
            - window: [bool] If True, delta also gives the minimal distance between peaks.

    Returns:
        movie_data: Class containing estimation data about all pixels
    """
    if method == "cond_av":
        assert {
            "min_threshold",
            "max_threshold",
            "delta",
            "window",
        } <= kwargs.keys(), (
            "Arguments must be provided: min_threshold, max_threshold, delta, window"
        )

    movie_data = MovieData(
        range(0, len(ds.x.values)),
        range(0, len(ds.y.values)),
        lambda i, j: estimate_velocities_for_pixel(i, j, ds, method, naive, **kwargs),
    )
    return movie_data
