import numpy as np
from abc import ABC
from typing import Tuple, Any
import xarray as xr
from numpy import ndarray, dtype


class ImagingDataInterface(ABC):
    def get_shape(self) -> Tuple[int, int]:
        """Returns a tuple (nr, nz) with the dimensions of the data in the r
        and z direction."""
        raise NotImplementedError

    def get_signal(self, x: int, y: int) -> np.ndarray:
        """Returns an array containing the time series at indexes x, y."""
        raise NotImplementedError

    def get_dt(self) -> float:
        """Returns time sampling time."""
        raise NotImplementedError

    def get_position(self, x: int, y: int) -> Tuple[float, float]:
        """Returns a tuple with the r and z coordinate of the data at index x,
        y."""
        raise NotImplementedError

    def is_pixel_dead(self, x: int, y: int) -> bool:
        """Returns True if the data at indexes x, y pertains a dead pixel and
        should not be used for estimation."""
        raise NotImplementedError

    def is_within_boundaries(self, x: int, y: int) -> bool:
        """Returns True if the indexes are valid."""
        shape = self.get_shape()
        return 0 <= x < shape[0] and 0 <= y < shape[1]


class CModImagingDataInterface(ImagingDataInterface):
    """Implementation of ImagingDataInterface for xarray datasets given by the
    code at https://github.com/sajidah-ahmed/cmod_functions."""

    def __init__(self, ds: xr.Dataset):
        self.ds = ds

    def get_shape(self) -> Tuple[int, int]:
        return self.ds.dims["x"], self.ds.dims["y"]

    def get_signal(self, x: int, y: int) -> np.ndarray:
        return self.ds.isel(x=x, y=y)["frames"].values

    def get_dt(self) -> float:
        times = self.ds["time"]
        return float(times[1].values - times[0].values)

    def get_position(self, x: int, y: int) -> Tuple[float, float]:
        return self.ds.R.isel(x=x, y=y).values, self.ds.Z.isel(x=x, y=y).values

    def is_pixel_dead(self, x: int, y: int) -> bool:
        signal = self.get_signal(x, y)
        return len(signal) == 0 or np.isnan(signal[0])


class SyntheticBlobImagingDataInterface(ImagingDataInterface):
    """Implementation of ImagingDataInterface for the datasets return by the
    code https://github.com/uit-cosmo/blobmodel."""

    def __init__(self, ds: xr.Dataset):
        self.ds = ds

    def get_shape(self) -> Tuple[int, int]:
        return self.ds.dims["x"], self.ds.dims["y"]

    def get_signal(self, x: int, y: int) -> np.ndarray:
        return self.ds.isel(x=x, y=y)["n"].values

    def get_dt(self) -> float:
        times = self.ds["t"]
        return float(times[1].values - times[0].values)

    def get_position(self, x: int, y: int) -> Tuple[float, float]:
        return self.ds.x.isel(x=x).values, self.ds.y.isel(y=y).values

    def is_pixel_dead(self, x: int, y: int) -> bool:
        signal = self.get_signal(x, y)
        return len(signal) == 0 or np.isnan(signal[0])
