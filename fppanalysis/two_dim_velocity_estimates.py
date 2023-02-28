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
        return ds.x.isel(x=x, y=y).values, ds.y.isel(x=x, y=y).values
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
        return ds.isel(x=x, y=y).dropna(dim="time", how="any")['frames'].values
    # 2d code
    if hasattr(ds, "t"):
        return ds.isel(x=x, y=y).dropna(dim="t", how="any")['n'].values
    raise "Unknown format"


def get_dt(ds):
    # Sajidah's format
    if hasattr(ds, "time"):
        times = ds['time']
        return times[1].values - times[0].values
    # 2d code
    if hasattr(ds, "t"):
        times = ds['t']
        return times[1].values - times[0].values
    raise "Unknown format"


def estimate_velocities_for_pixel(x, y, ds):
    """
    Estimates radial and poloidal velocity for a pixel with indexes x,y using the pixels located
    at (x,y), (x+1, y) and (x, y-1). Time delay estimation is performed by maximizing the cross-
    correlation function.

    Returns:
        vx Radial velocity
        vy Poloidal velocity
    """
    x_down, y_down = x, y - 1
    x_right, y_right = x + 1, y
    dt = get_dt(ds)

    R0, Z0 = get_rz(x, y, ds)
    R_right, Z_right = get_rz(x_right, y_right, ds)
    R_down, Z_down = get_rz(x_down, y_down, ds)
    frames_0 = get_signal(x, y, ds)
    frames_right = get_signal(x_right, y_right, ds)
    frames_down = get_signal(x_down, y_down, ds)
    if len(frames_0) == 0 or len(frames_right) == 0 or len(frames_down) == 0:
        return 0, 0

    delta_vert = abs(Z0 - Z_down)
    delta_tvert = tde.estimate_time_delay_ccmax(x=frames_0, y=frames_down, dt=dt)
    if Z0 < Z_down:
        delta_tvert = -delta_tvert

    delta_hor = abs(R_right - R0)
    delta_thor = tde.estimate_time_delay_ccmax(x=frames_right, y=frames_0, dt=dt)

    vx, vy = get_2d_velocities_from_time_delays(delta_thor, delta_tvert, delta_hor, delta_vert)

    return vx, vy


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

    for i in range(0, shape[0] - 1):
        for j in range(1, shape[1]):
            try:
                vx[i, j], vy[i, j] = estimate_velocities_for_pixel(i, j, ds)
            except:
                print("Issues estimating velocity for pixel", i, j,
                      "Run estimate_velocities_for_pixel(i, j, ds) to get a detailed error stacktrace")

    vx[np.isnan(vx) | np.isinf(vx)] = 0
    vy[np.isnan(vy) | np.isinf(vy)] = 0
    return vx, vy, R, Z
