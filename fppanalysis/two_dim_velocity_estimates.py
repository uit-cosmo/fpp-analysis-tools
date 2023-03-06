import fppanalysis.time_delay_estimation as tde
import numpy as np


def get_2d_velocities_from_time_delays(delta_tx, delta_ty, delta_x, delta_y):
    """
    Estimates radial and poloidal velocities given the input parameters:
    Input:
         delta_tx Estimation of the time delay between radially separated points.
         delta_ty Estimation of the time delay between poloidally separated points.
         delta_x Spatial separation between radially separated points.
         delta_y Spatial separation between poloidally separated points.

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


def get_rz(x, y, ds):
    # Sajidah's format
    if hasattr(ds, "time"):
        return ds.R.isel(x=x, y=y).values, ds.Z.isel(x=x, y=y).values
    # 2d code
    if hasattr(ds, "t"):
        return ds.x.isel(x=x).values, ds.y.isel(y=y).values
    raise "Unknown format"


def get_rz_full(ds):
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


def get_signal(x, y, ds):
    # Sajidah's format
    if hasattr(ds, "time"):
        return ds.isel(x=x, y=y).dropna(dim="time", how="any")["frames"].values
    # 2d code
    if hasattr(ds, "t"):
        return ds.isel(x=x, y=y).dropna(dim="t", how="any")["n"].values
    raise "Unknown format"


def get_dt(ds):
    # Sajidah's format
    if hasattr(ds, "time"):
        times = ds["time"]
        return times[1].values - times[0].values
    # 2d code
    if hasattr(ds, "t"):
        times = ds["t"]
        return times[1].values - times[0].values
    raise "Unknown format"


def estimate_velocities_given_points(p0, p1, p2, ds):
    dt = get_dt(ds)
    r0, z0 = get_rz(p0[0], p0[1], ds)
    r1, z1 = get_rz(p1[0], p1[1], ds)
    r2, z2 = get_rz(p2[0], p2[1], ds)
    signal0 = get_signal(p0[0], p0[1], ds)
    signal1 = get_signal(p1[0], p1[1], ds)
    signal2 = get_signal(p2[0], p2[1], ds)

    if len(signal0) == 0 or len(signal1) == 0 or len(signal2) == 0:
        return None

    delta_tz = tde.estimate_time_delay_ccmax(x=signal2, y=signal0, dt=dt)
    delta_tx = tde.estimate_time_delay_ccmax(x=signal1, y=signal0, dt=dt)

    return get_2d_velocities_from_time_delays(delta_tx, delta_tz, r1 - r0, z2 - z0)


def is_within_boundaries(p, ds):
    return 0 <= p[0] < ds.sizes["x"] and 0 <= p[1] < ds.sizes["y"]


def estimate_velocities_for_pixel(x, y, ds):
    """Estimates radial and poloidal velocity for a pixel with indexes x,y
    using all four possible combinations of nearest neighbour pixels (x-1, y),
    (x, y+1), (x+1, y) and (x, y-1). Dead-pixels (stored as np.nan arrays) are
    ignored. Pixels outside the coordinate domain are ignored. Time delay
    estimation is performed by maximizing the cross- correlation function.

    Returns:
        vx Radial velocity
        vy Poloidal velocity
    """

    h_neighbors = [(x - 1, y), (x + 1, y)]
    v_neighbors = [(x, y - 1), (x, y + 1)]
    results = [
        estimate_velocities_given_points((x, y), px, py, ds)
        for px in h_neighbors
        if is_within_boundaries(px, ds)
        for py in v_neighbors
        if is_within_boundaries(py, ds)
    ]
    results = [r for r in results if r is not None]
    if len(results) == 0:  # If (x,y) is dead we cannot estimate
        return None, None
    mean_vx = sum(map(lambda r: r[0], results)) / len(results)
    mean_vy = sum(map(lambda r: r[1], results)) / len(results)
    return mean_vx, mean_vy


def estimate_velocity_field(ds):
    """
    Given a dataset ds with GPI data in a format produced by https://github.com/sajidah-ahmed/cmod_functions,
    computed the velocity field. The estimation takes into account poloidal flows as described in
    the 2D filament model. For each pixel, the velocities are estimated using the given pixel, and two neighbour
    pixels: the right neighbour and the down neighbour. The return objects are matrices of the size of the
    GPI grid, from which the velocity field can be easily plotted via f.e matplotlib.quiver.

    Returns:
        vx Radial velocities
        vy Poloidal velocities
        R Radial positions
        Z Radial positions
    """
    shape = (len(ds.x.values), len(ds.y.values))
    vx = np.zeros(shape=shape)
    vy = np.zeros(shape=shape)
    R, Z = get_rz_full(ds)

    for i in range(0, shape[0]):
        for j in range(0, shape[1]):
            try:
                vx[i, j], vy[i, j] = estimate_velocities_for_pixel(i, j, ds)
            except:
                print(
                    "Issues estimating velocity for pixel",
                    i,
                    j,
                    "Run estimate_velocities_for_pixel(i, j, ds) to get a detailed error stacktrace",
                )

    vx[np.isnan(vx) | np.isinf(vx)] = 0
    vy[np.isnan(vy) | np.isinf(vy)] = 0
    return vx, vy, R, Z
