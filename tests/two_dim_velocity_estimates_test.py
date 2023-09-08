import numpy as np
from blobmodel import Model, DefaultBlobFactory
import fppanalysis.two_dim_velocity_estimates as td
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


def test_rad_and_pol():
    v, w = 1, 1
    ds = make_2d_realization(v, w, np.array([5, 6, 7]), np.array([5, 6, 7]))
    pd = td.estimate_velocities_for_pixel(1, 1, ds)
    v_est, w_est, = (
        pd.vx,
        pd.vy,
    )
    error = np.max([abs(v_est - v), abs(w_est - w)])
    assert error < 0.1, "Numerical error too big"


def test_full():
    v, w = 1, 1
    ds = make_2d_realization(v, w, np.array([5, 6, 7, 8]), np.array([5, 6, 7, 8]))
    movie_data = td.estimate_velocity_field(ds)
    vx = movie_data.get_vx()
    assert np.max(np.abs(vx - np.ones(shape=(4, 4)))) < 0.1, "Numerical error too big"


def test_rad_and_neg_pol():
    v, w = 1, -1
    ds = make_2d_realization(v, w, np.array([5, 6, 7]), np.array([5, 6, 7]))
    pd = td.estimate_velocities_for_pixel(1, 1, ds)
    v_est, w_est, = (
        pd.vx,
        pd.vy,
    )
    error = np.max([abs(v_est - v), abs(w_est - w)])
    assert error < 0.1, "Numerical error too big"


def test_rad_and_2pol():
    v, w = 1, 2
    ds = make_2d_realization(v, w, np.array([5, 6, 7]), np.array([5, 6, 7]))
    pd = td.estimate_velocities_for_pixel(1, 1, ds)
    v_est, w_est, = (
        pd.vx,
        pd.vy,
    )
    error = np.max([abs(v_est - v), abs(w_est - w)])
    assert error < 0.1, "Numerical error too big"


def test_cond_av():
    v, w = 1, -1
    ds = make_2d_realization(v, w, np.array([5, 6, 7]), np.array([5, 6, 7]))
    method = "cond_av"
    min_threshold = 2.5
    max_threshold = np.inf
    delta = 5
    window = True
    pd = td.estimate_velocities_for_pixel(
        1,
        1,
        ds,
        method=method,
        min_threshold=min_threshold,
        max_threshold=max_threshold,
        delta=delta,
        window=window,
    )
    v_est, w_est, = (
        pd.vx,
        pd.vy,
    )
    error = np.max([abs(v_est - v), abs(w_est - w)])
    assert error < 0.1, "Numerical error too big"


# Dead pixels have already been preprocessed and have an array of nans at their site
class MockXDS(xr.Dataset):
    def __init__(self, ds):
        super().__init__(ds)
        self.ds = ds

    def isel(
        self,
        indexers=None,
        drop: bool = False,
        missing_dims: str = "raise",
        **indexers_kwargs
    ):
        if indexers_kwargs["x"] == 0:
            dummy = np.array([np.nan, np.nan, np.nan])
            return xr.Dataset({"n": (["t"], dummy)})
        return self.ds.isel(indexers, drop, missing_dims, **indexers_kwargs)


def test_ignore_dead_pixels():
    v, w = 1, 1
    ds = make_2d_realization(v, w, np.array([5, 6, 7]), np.array([5, 6, 7]))
    mock_ds = MockXDS(ds)
    pd = td.estimate_velocities_for_pixel(1, 1, mock_ds)
    v_est, w_est, = (
        pd.vx,
        pd.vy,
    )
    error = np.max([abs(v_est - v), abs(w_est - w)])
    assert error < 0.1, "Numerical error too big"


def test_interpolate():
    v, w = 1.2, 1
    # Without interpolation, there is no enough time resolution (dt = 0.1) to find the cross-correlation maximum
    ds = make_2d_realization(v, w, np.array([1, 1.1]), np.array([5, 5.1]), dt=0.1)
    pd = td.estimate_velocities_for_pixel(1, 1, ds, method="cross_corr_interpolate")
    v_est, w_est, = (
        pd.vx,
        pd.vy,
    )
    error = np.max([abs(v_est - v), abs(w_est - w)])
    assert error < 0.1, "Numerical error too big"
