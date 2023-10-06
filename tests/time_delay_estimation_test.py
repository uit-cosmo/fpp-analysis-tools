import numpy as np

import fppanalysis.time_delay_estimation as tde
import fppanalysis.utils as u
import test_utils as tu


def test_interpolate():
    v, w = 1, 0
    dt = 0.2
    # Without interpolation, there is no enough time resolution (dt = 0.1) to find the cross-correlation maximum
    ds = tu.make_2d_realization(
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
