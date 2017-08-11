"""
Collection of functions used in pvlib_python
"""

import datetime as dt

import numpy as np
import pandas as pd
import xarray as xr
import pytz


def cosd(angle):
    """
    Cosine with angle input in degrees

    Parameters
    ----------
    angle : float or array-like
        Angle in degrees

    Returns
    -------
    result : float or array-like
        Cosine of the angle
    """

    res = xr.ufuncs.cos(xr.ufuncs.deg2rad(angle))
    return res


def sind(angle):
    """
    Sine with angle input in degrees

    Parameters
    ----------
    angle : float
        Angle in degrees

    Returns
    -------
    result : float
        Sin of the angle
    """

    res = xr.ufuncs.sin(xr.ufuncs.deg2rad(angle))
    return res


def tand(angle):
    """
    Tan with angle input in degrees

    Parameters
    ----------
    angle : float
        Angle in degrees

    Returns
    -------
    result : float
        Tan of the angle
    """

    res = xr.ufuncs.tan(xr.ufuncs.deg2rad(angle))
    return res


def asind(number):
    """
    Inverse Sine returning an angle in degrees

    Parameters
    ----------
    number : float
        Input number

    Returns
    -------
    result : float
        arcsin result
    """

    res = xr.ufuncs.rad2deg(xr.ufuncs.arcsin(number))
    return res


def localize_to_utc(time, location):
    """
    Converts or localizes a time series to UTC.

    Parameters
    ----------
    time : datetime.datetime, pandas.DatetimeIndex,
           or pandas.Series/DataFrame with a DatetimeIndex.
    location : pvlib.Location object

    Returns
    -------
    pandas object localized to UTC.
    """
    import datetime as dt
    import pytz

    if isinstance(time, dt.datetime):
        if time.tzinfo is None:
            time = pytz.timezone(location.tz).localize(time)
        time_utc = time.astimezone(pytz.utc)
    else:
        try:
            time_utc = time.tz_convert('UTC')
        except TypeError:
            time_utc = time.tz_localize(location.tz).tz_convert('UTC')

    return time_utc


def datetime_to_djd(time):
    """
    Converts a datetime to the Dublin Julian Day

    Parameters
    ----------
    time : datetime.datetime
        time to convert

    Returns
    -------
    float
        fractional days since 12/31/1899+0000
    """

    if time.tzinfo is None:
        time_utc = pytz.utc.localize(time)
    else:
        time_utc = time.astimezone(pytz.utc)

    djd_start = pytz.utc.localize(dt.datetime(1899, 12, 31, 12))
    djd = (time_utc - djd_start).total_seconds() * 1.0/(60 * 60 * 24)

    return djd


def djd_to_datetime(djd, tz='UTC'):
    """
    Converts a Dublin Julian Day float to a datetime.datetime object

    Parameters
    ----------
    djd : float
        fractional days since 12/31/1899+0000
    tz : str, default 'UTC'
        timezone to localize the result to

    Returns
    -------
    datetime.datetime
       The resultant datetime localized to tz
    """

    djd_start = pytz.utc.localize(dt.datetime(1899, 12, 31, 12))

    utc_time = djd_start + dt.timedelta(days=djd)
    return utc_time.astimezone(pytz.timezone(tz))


def _pandas_to_doy(pd_object):
    """
    Finds the day of year for a pandas datetime-like object.

    Useful for delayed evaluation of the dayofyear attribute.

    Parameters
    ----------
    pd_object : DatetimeIndex or Timestamp

    Returns
    -------
    dayofyear
    """
    return pd_object.dayofyear


def _doy_to_datetimeindex(doy, epoch_year=2014):
    """
    Convert a day of year scalar or array to a pd.DatetimeIndex.

    Parameters
    ----------
    doy : numeric
        Contains days of the year

    Returns
    -------
    pd.DatetimeIndex
    """
    doy = np.atleast_1d(doy).astype('float')
    epoch = pd.Timestamp('{}-12-31'.format(epoch_year - 1))
    timestamps = [epoch + dt.timedelta(days=adoy) for adoy in doy]
    return pd.DatetimeIndex(timestamps)


def _datetimelike_scalar_to_doy(time):
    return pd.Timestamp(time).timetuple().tm_yday


def _datetimelike_scalar_to_datetimeindex(time):
    return pd.DatetimeIndex([pd.Timestamp(time)])


def _bool_false_is_nan(cond, true_values=1):
    return (0 / cond) + true_values


def dt_index_to_xr(time):
    return np.array(time, dtype=np.datetime64)


def xr_to_dt_index(time):
    time = time.astype("datetime64[ns]")
    return time.coords.to_index() if time.coords else pd.DatetimeIndex(time.values, name=time.dims[0])


def recursive_call_multi_locations(time, latitude, longitude, func, **kwargs):
    # Recursive call to func if latitude and longitude are xarrays
    if isinstance(latitude, xr.DataArray) and isinstance(longitude, xr.DataArray):

        try:
            if max(latitude.ndim, longitude.ndim) != 1:
                raise ValueError("Latitude or Longitude have multiple dimensions.")
            xr.testing.assert_equal(latitude.coords.to_dataset(), longitude.coords.to_dataset())
            spatial_dim = latitude.coords.to_index()
        except AssertionError as ae:
            raise ValueError("Latitude and Longitude do not have equal coordinates in.")

        if isinstance(time,pd.DatetimeIndex):
            time_dim = time.name if time.name else "index"
            time_index = time
            time_array = dt_index_to_xr(time)
        elif isinstance(time,xr.DataArray):
            time_array = time
            time_index = xr_to_dt_index(time)
            time_dim = time_index.name if time_index.name else "index"
        else:
            raise ValueError("time must be either DateTimeIndex or a xarray.DataArray")

        def recursive_func(lat, lon):
            single = func(time_index, lat, lon, **kwargs)
            if isinstance(single,pd.Series):
                single = pd.DataFrame(single)
            single_ds = xr.Dataset.from_dataframe(single).to_array(dim="columns")
            single_ds[time_dim] = time_array
            return single_ds

        output = xr.concat([recursive_func(lat.values.item(), lon.values.item()) for lat, lon in zip(latitude, longitude)], dim=spatial_dim)

        if output.sizes["columns"] == 1:
            return output.squeeze("columns",drop=True)
        else:
            return output.to_dataset(dim="columns")

    else:
        raise ValueError("Recursive calls can only be made on xarray.DataArray inputs for latitude and longitude.")


def _scalar_out(input):
    if np.isscalar(input):
        output = input
    else:  #
        # works if it's a 1 length array and
        # will throw a ValueError otherwise
        output = np.asscalar(input)

    return output


def _array_out(input):
    if isinstance(input, pd.Series):
        output = input.values
    elif isinstance(input, xr.DataArray):
        output = input.values
    else:
        output = input

    return output


def _build_kwargs(keys, input_dict):
    """
    Parameters
    ----------
    keys : iterable
        Typically a list of strings.
    adict : dict-like
        A dictionary from which to attempt to pull each key.

    Returns
    -------
    kwargs : dict
        A dictionary with only the keys that were in input_dict
    """

    kwargs = {}
    for key in keys:
        try:
            kwargs[key] = input_dict[key]
        except KeyError:
            pass

    return kwargs
