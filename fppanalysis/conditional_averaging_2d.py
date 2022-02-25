from itertools import product
import numpy as np


def condavg_2d(ts_1, ts_2, ts_avg, edges_ts1, edges_ts2):
    """
    Computes the average of ts_avg
    of density and temperature fluctuations
    Input:
    ======
    ts_1.......: ndarray, float: Time series 1
    ts_2.......: ndarray, float: Time series 2
    ts_avg.....: ndarray, float: Time series to average within bins
    edges_ts1..: ndarray, float: Bin edges for ts_1
    edges_ts2..: ndarray, float: Bin edges for ts_2

    Output:
    =======
    vrad_avg.: ndarray, float: Average value of velocity fluctuations
    """

    assert len(ts_1.shape) == 1
    assert ts_1.shape == ts_2.shape
    assert ts_1.shape == ts_avg.shape

    assert edges_ts1[0] < edges_ts1[-1]
    assert edges_ts2[0] < edges_ts2[-1]

    avg = np.zeros([edges_ts1.shape[0] - 1, edges_ts2.shape[0] - 1], dtype="float64")

    ts1_idx_list = []
    ts2_idx_list = []

    ts1_idx_rg = np.arange(edges_ts1.shape[0] - 1)
    ts2_idx_rg = np.arange(edges_ts2.shape[0] - 1)

    for t in ts1_idx_rg:
        ts1_idx_list.append(
            set(
                np.argwhere((ts_1 > edges_ts1[t]) & (ts_1 < edges_ts1[t + 1])).flatten()
            )
        )
    for t in ts2_idx_rg:
        # Get ne indices where fluctuations are in between current edges
        ts2_idx_list.append(
            set(
                np.argwhere((ts_2 > edges_ts2[t]) & (ts_2 < edges_ts2[t + 1])).flatten()
            )
        )

    for t1_idx, t2_idx in product(ts1_idx_rg, ts2_idx_rg):
        avg_idx = ts1_idx_list[t1_idx].intersection(ts2_idx_list[t2_idx])
        #    # Assign average velocity if we have elements in the intersection
        if avg_idx:
            avg[t1_idx, t2_idx] = ts_avg[list(avg_idx)].mean()

    return avg

