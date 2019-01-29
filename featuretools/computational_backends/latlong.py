import numpy as np
import pandas as pd
from pandas.arrays import PandasArray
from pandas.api.extensions import ExtensionDtype, ExtensionArray, register_extension_dtype


class LatLongDtypeType(type):
    pass


@register_extension_dtype
class LatLongDtype(ExtensionDtype):
    type = LatLongDtypeType

    @property
    def name(self):
        return "latlong"

    @classmethod
    def construct_from_string(cls, string):
        if string == cls.name:
            return cls()
        else:
            msg = "Cannot construct a '{}' from '{}'".format(cls, string)
            raise TypeError(msg)


class LatLongArray(ExtensionArray):
    def __init__(self, latitude, longitude):
        self.latitude = pd.Series(latitude)
        self.longitude = pd.Series(longitude)

    @classmethod
    def _from_sequence(cls, scalars, dtype=None, copy=False):
        latitude = np.array([scalar[0] for scalar in scalars])
        longitude = np.array([scalar[1] for scalar in scalars])
        return LatLongArray(latitude, longitude)

    @classmethod
    def _from_factorized(cls, values, original):
        raise NotImplementedError

    def __getitem__(self, item):
        if isinstance(item, int):
            return (self.latitude[item], self.longitude[item])
        else:
            latitude = self.latitude[item]
            longitude = self.longitude[item]
            return LatLongArray(latitude, longitude)

    def __len__(self):
        return len(self.latitude)

    @property
    def dtype(self):
        return LatLongDtype()

    @property
    def nbytes(self):
        return self.latitude.nbytes + self.longitude.nbytes

    def isna(self):
        return (self.latitude.isna() & self.longitude.isna()).values

    def take(self, indices, allow_fill=False, fill_value=None):
        lat_result = self.latitude.take(indices)
        long_result = self.longitude.take(indices)
        return LatLongArray(lat_result, long_result)

    def copy(self, deep=False):
        if deep:
            return LatLongArray(self.latitude.copy(), self.longitude.copy())
        return LatLongArray(self.latitude, self.longitude)

    @classmethod
    def _concat_same_type(cls, to_concat):
        latitude = PandasArray._concat_same_type([arr.latitude.array for arr in to_concat])
        longitude = PandasArray._concat_same_type([arr.longitude.array for arr in to_concat])
        return LatLongArray(latitude, longitude)
