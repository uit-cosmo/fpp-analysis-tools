import numpy as np
from blobmodel import Model, DefaultBlobFactory

import fppanalysis.time_delay_estimation as tde
import fppanalysis.utils as u
import xarray as xr


def make_2d_realization(
    vx, vy, xpoints, ypoints, T=1000, K=1000, Lx=10, Ly=10, taup=1e10, dt=0.01
):
    bf = DefaultBlobFactory(
        A_dist="deg",
        vx_dist="deg",
        vy_dist="deg",
        vy_parameter=vy,
        vx_parameter=vx,
    )
    bm = Model(
        Nx=10,
        Ny=1,
        Lx=Lx,
        Ly=Ly,
        dt=dt,
        T=T,
        num_blobs=K,
        blob_shape="gauss",
        periodic_y=True,
        t_drain=taup,
        blob_factory=bf,
    )
    update_geometry(xpoints, ypoints, bm)
    return bm.make_realization(speed_up=True, error=10e-2)


# Change the geometry of the 2d model to lie similar to APD data.
def update_geometry(x_grid, y_grid, model):
    x_matrix, y_matrix, t_matrix = np.meshgrid(x_grid, y_grid, model._geometry.t)
    model._geometry.x_matrix = x_matrix
    model._geometry.y_matrix = y_matrix
    model._geometry.t_matrix = t_matrix
    model._geometry.Ny = len(y_grid)
    model._geometry.Nx = len(x_grid)
    model._geometry.x = x_grid
    model._geometry.y = y_grid


def test_interpolate():
    v, w = 1, 0
    dt = 0.2
    # Without interpolation, there is no enough time resolution (dt = 0.1) to find the cross-correlation maximum
    ds = make_2d_realization(
        v, w, np.array([1, 1.1]), np.array([5, 5.1]), dt=dt, K=1000, T=1000
    )
    x = u.get_signal(0, 0, ds)
    y = u.get_signal(1, 0, ds)

    td_interpolate, _, _ = tde.estimate_time_delay_ccmax_running_mean(
        y, x, dt, tde.CCOptions(cc_window=100, interpolate=True)
    )
    td_no_interpolate, _, _ = tde.estimate_time_delay_ccmax_running_mean(
        y, x, dt, tde.CCOptions(cc_window=100, interpolate=False)
    )

    expected = 0.1
    expected_no_interpolate = 0

    assert np.abs(td_interpolate - expected) < 0.05, "Numerical error too big"
    assert (
        np.abs(td_no_interpolate - expected_no_interpolate) < 0.05
    ), "Numerical error too big"


def test_local_maxima():
    x = np.array([0, 1, 0, 1, 0])
    assert tde._count_local_maxima(x) == 2


def test_local_maxima_only_if_big_enough():
    x = np.array([0, 1, 0, 0.4, 0])
    assert tde._count_local_maxima(x) == 1
