import numpy as np
import fppanalysis.time_delay_estimation as tde
import fppanalysis.two_dim_velocity_estimates as td
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
    pd = td.estimate_velocities_for_pixel(1, 1, ds, estimation_options)
    (
        v_est,
        w_est,
    ) = (
        pd.vx,
        pd.vy,
    )
    error = np.max([abs(v_est - v), abs(w_est - w)])
    assert error < 0.1, "Numerical error too big"


def test_full():
    v, w = 1, 1
    ds = tu.make_2d_realization(v, w, np.array([5, 6, 7]), np.array([5, 6, 7, 8]))
    eo = get_estimation_options()
    movie_data = td.estimate_velocity_field(ds, eo)
    vx = movie_data.get_vx()
    assert np.max(np.abs(vx - np.ones(shape=(4, 3)))) < 0.1, "Numerical error too big"


def test_rad_and_neg_pol():
    v, w = 1, -1
    ds = tu.make_2d_realization(v, w, np.array([5, 6, 7]), np.array([5, 6, 7]))
    pd = td.estimate_velocities_for_pixel(1, 1, ds, get_estimation_options())
    (
        v_est,
        w_est,
    ) = (
        pd.vx,
        pd.vy,
    )
    error = np.max([abs(v_est - v), abs(w_est - w)])
    assert error < 0.1, "Numerical error too big"


def test_rad_and_2pol():
    v, w = 1, 2
    ds = tu.make_2d_realization(v, w, np.array([5, 6, 7]), np.array([5, 6, 7]))
    pd = td.estimate_velocities_for_pixel(1, 1, ds, get_estimation_options())
    (
        v_est,
        w_est,
    ) = (
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
    pd = td.estimate_velocities_for_pixel(1, 1, ds, estimation_options)
    (
        v_est,
        w_est,
    ) = (
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
    pd = td.estimate_velocities_for_pixel(1, 1, mock_ds, get_estimation_options())
    (
        v_est,
        w_est,
    ) = (
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
    pd = td.estimate_velocities_for_pixel(0, 0, ds, estimation_options)
    (
        v_est,
        w_est,
    ) = (
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
    pd = td.estimate_velocities_for_pixel(1, 1, ds, eo)
    (
        v_est,
        w_est,
    ) = (
        pd.vx,
        pd.vy,
    )
    error = np.max([abs(v_est - v), abs(w_est - w)])
    assert error < 0.1, "Numerical error too big"
