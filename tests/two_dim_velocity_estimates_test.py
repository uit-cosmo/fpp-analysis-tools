import numpy as np
import fppanalysis.time_delay_estimation as tde
import fppanalysis.two_dim_velocity_estimates as td
import fppanalysis.utils as u
import test_utils as tu
import xarray as xr


def get_estimation_options():
    return td.EstimationOptions(
        method=tde.TDEMethod.CC,
        use_3point_method=True,
    )


def test_rad_and_pol():
    v, w = 1, 1
    ds = tu.make_2d_realization(v, w, np.array([5, 6, 7]), np.array([5, 6, 7]))
    estimation_options = get_estimation_options()
    pd = td.estimate_velocities_for_pixel(
        1, 1, u.SyntheticBlobImagingDataInterface(ds), estimation_options
    )
    (v_est, w_est,) = (
        pd.vx,
        pd.vy,
    )
    error = np.max([abs(v_est - v), abs(w_est - w)])
    assert error < 0.1, "Numerical error too big"


def test_full():
    v, w = 1, 1
    ds = tu.make_2d_realization(v, w, np.array([5, 6, 7]), np.array([5, 6, 7, 8]))
    eo = get_estimation_options()
    movie_data = td.estimate_velocity_field(u.SyntheticBlobImagingDataInterface(ds), eo)
    vx = movie_data.get_vx()
    assert np.max(np.abs(vx - np.ones(shape=(4, 3)))) < 0.1, "Numerical error too big"


def test_rad_and_neg_pol():
    v, w = 1, -1
    ds = tu.make_2d_realization(v, w, np.array([5, 6, 7]), np.array([5, 6, 7]))
    pd = td.estimate_velocities_for_pixel(
        1, 1, u.SyntheticBlobImagingDataInterface(ds), get_estimation_options()
    )
    (v_est, w_est,) = (
        pd.vx,
        pd.vy,
    )
    error = np.max([abs(v_est - v), abs(w_est - w)])
    assert error < 0.1, "Numerical error too big"


def test_rad_and_2pol():
    v, w = 1, 2
    ds = tu.make_2d_realization(v, w, np.array([5, 6, 7]), np.array([5, 6, 7]))
    pd = td.estimate_velocities_for_pixel(
        1, 1, u.SyntheticBlobImagingDataInterface(ds), get_estimation_options()
    )
    (v_est, w_est,) = (
        pd.vx,
        pd.vy,
    )
    error = np.max([abs(v_est - v), abs(w_est - w)])
    assert error < 0.1, "Numerical error too big"


def test_cond_av():
    v, w = 1, -1
    ds = tu.make_2d_realization(v, w, np.array([5, 6, 7]), np.array([5, 6, 7]))
    estimation_options = get_estimation_options()
    cond_av_eo = tde.CAOptions(delta=5, window=True)
    estimation_options.method = tde.TDEMethod.CA
    estimation_options.ca_options = cond_av_eo
    pd = td.estimate_velocities_for_pixel(
        1, 1, u.SyntheticBlobImagingDataInterface(ds), estimation_options
    )
    (v_est, w_est,) = (
        pd.vx,
        pd.vy,
    )
    error = np.max([abs(v_est - v), abs(w_est - w)])
    assert error < 0.15, "Numerical error too big"


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
    ds = tu.make_2d_realization(v, w, np.array([5, 6, 7]), np.array([5, 6, 7]))
    mock_ds = MockXDS(ds)
    pd = td.estimate_velocities_for_pixel(
        1, 1, u.SyntheticBlobImagingDataInterface(mock_ds), get_estimation_options()
    )
    (v_est, w_est,) = (
        pd.vx,
        pd.vy,
    )
    error = np.max([abs(v_est - v), abs(w_est - w)])
    assert error < 0.1, "Numerical error too big"


def test_neighbours():
    v, w = 1.05, 1.05
    ds = tu.make_2d_realization(
        v, w, np.array([5, 5.1, 5.2]), np.array([5, 5.1, 5.2]), dt=0.1
    )
    estimation_options = get_estimation_options()
    estimation_options.neighbour_options.ccf_min_lag = 1
    estimation_options.neighbour_options.max_separation = 3
    pd = td.estimate_velocities_for_pixel(
        0, 0, u.SyntheticBlobImagingDataInterface(ds), estimation_options
    )
    (v_est, w_est,) = (
        pd.vx,
        pd.vy,
    )
    error = np.max([abs(v_est - v), abs(w_est - w)])
    assert error < 0.1, "Numerical error too big"


def test_cross_corr_fit():
    v, w = 1, 1
    ds = tu.make_2d_realization(v, w, np.array([5, 6, 7]), np.array([5, 6, 7]))
    eo = get_estimation_options()
    eo.method = tde.TDEMethod.CCFit
    pd = td.estimate_velocities_for_pixel(
        1, 1, u.SyntheticBlobImagingDataInterface(ds), eo
    )
    (v_est, w_est,) = (
        pd.vx,
        pd.vy,
    )
    error = np.max([abs(v_est - v), abs(w_est - w)])
    assert error < 0.1, "Numerical error too big"


def test_non_orthogonal_points():
    from blobmodel import Model, DefaultBlobFactory

    T = 1000
    K = 1000
    Lx = 10
    Ly = 10
    taup = 1e10
    dt = 0.01

    vx, vy = 1, 1

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

    times = np.arange(0, T, dt)
    x = np.array([0.0, 1.0])
    y = np.array([5.0, 6.0])

    x_matrix, y_matrix, t_matrix = np.meshgrid(x, y, times)
    x_matrix[1, 0, :] = 1.5
    x_matrix[1, 1, :] = 2.5

    bm._geometry.x_matrix = x_matrix
    bm._geometry.y_matrix = y_matrix
    bm._geometry.t_matrix = t_matrix
    bm._geometry.Ny = len(y)
    bm._geometry.Nx = len(x)
    bm._geometry.x = x
    bm._geometry.y = y

    ds = bm.make_realization(speed_up=True, error=10e-2)

    ds_new = xr.Dataset(
        {"frames": (["y", "x", "time"], ds.n.values)},
        coords={
            "R": (["y", "x"], x_matrix[:, :, 0]),
            "Z": (["y", "x"], y_matrix[:, :, 0]),
            "time": (["time"], times),
        },
    )

    estimation_options = get_estimation_options()
    estimation_options.cc_options.cc_window = 1000

    movie_data = td.estimate_velocity_field(
        u.CModImagingDataInterface(ds_new), estimation_options
    )
    vx = movie_data.get_vx()
    assert np.max(np.abs(vx - np.ones(shape=(2, 2)))) < 0.1, "Numerical error too big"
