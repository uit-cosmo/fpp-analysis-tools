import numpy as np


def get_rz(x, y, ds):
    """
    Get R and Z values for an index
    """
    # Exp format
    if hasattr(ds, "time"):
        return ds.R.isel(x=x, y=y).values, ds.Z.isel(x=x, y=y).values
    # 2d code
    if hasattr(ds, "t"):
        return ds.x.isel(x=x).values, ds.y.isel(y=y).values
    raise "Unknown format"


def get_rz_full(ds):
    """
    Get all R and Z values
    """
    # Exp format
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
    """
    Get signal at a given indexes.
    """
    # Exp format
    if hasattr(ds, "time"):
        # return ds.isel(x=x, y=y).dropna(dim="time", how="any")["frames"].values
        return ds.isel(x=x, y=y)["frames"].values
    # 2d code
    if hasattr(ds, "t"):
        return ds.isel(x=x, y=y)["n"].values
    raise "Unknown format"


def get_dt(ds):
    """
    Get sampling time
    """
    # Exp format
    if hasattr(ds, "time"):
        times = ds["time"]
        return times[1].values - times[0].values
    # 2d code
    if hasattr(ds, "t"):
        times = ds["t"]
        return times[1].values - times[0].values
    raise "Unknown format"


def get_time(x, y, ds):
    """
    Get time array for given indexes
    """
    # Exp format
    if hasattr(ds, "time"):
        return ds.isel(x=x, y=y).time.values
    # 2d code
    if hasattr(ds, "t"):
        return ds.t.values
    raise "Unknown format"


def is_pixel_dead(x):
    """
    Returns True if the length of the signal is 0, or if the first element is np.nan
    """
    return len(x) == 0 or np.isnan(x[0])


def is_within_boundaries(p, ds):
    """
    Returns True if the tuple p represents indexes within the ranges of the dataset.
    """
    return 0 <= p[0] < ds.sizes["x"] and 0 <= p[1] < ds.sizes["y"]
