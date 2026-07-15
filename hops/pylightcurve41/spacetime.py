"""Offline-safe coordinate and time helpers used by the fitting core.

The upstream PyLightcurve compatibility layer imports the top-level ExoClock
package for these conversions.  Recent ExoClock/Astroquery combinations query
SIMBAD while that package is imported, which makes an otherwise local fit
depend on network availability.  Astropy provides the same barycentric and
heliocentric corrections without importing the catalogue client.
"""

from __future__ import annotations

import numpy as np
import astropy.units as u
from astropy.coordinates import Angle, EarthLocation, SkyCoord
from astropy.time import Time

from .errors import PyLCInputError


_GEOCENTER = EarthLocation.from_geocentric(0.0, 0.0, 0.0, unit=u.m)


class Degrees:
    """Small ExoClock-compatible degree wrapper for legacy PyLightcurve callers."""

    def __init__(self, degrees, arcminutes=0.0, arcseconds=0.0):
        if arcminutes or arcseconds:
            sign = -1.0 if float(degrees) < 0 else 1.0
            value = float(degrees) + sign * (
                float(arcminutes) / 60.0 + float(arcseconds) / 3600.0
            )
            self._degrees = value
        else:
            self._degrees = float(Angle(degrees, unit=u.deg).degree)

    def deg(self) -> float:
        return self._degrees % 360.0

    def deg_coord(self) -> float:
        return float(Angle(self._degrees * u.deg).wrap_at(180 * u.deg).degree)


class Hours:
    """Small ExoClock-compatible hour-angle wrapper for legacy callers."""

    def __init__(self, hours, minutes=0.0, seconds=0.0):
        if minutes or seconds:
            value = float(hours) + float(minutes) / 60.0 + float(seconds) / 3600.0
            self._degrees = value * 15.0
        else:
            self._degrees = float(Angle(hours, unit=u.hourangle).degree)

    def deg(self) -> float:
        return self._degrees % 360.0


def _values(time_array) -> tuple[np.ndarray, bool]:
    scalar = np.isscalar(time_array)
    try:
        values = np.atleast_1d(np.asarray(time_array, dtype=float))
    except (TypeError, ValueError) as exc:
        raise PyLCInputError("The time array could not be converted to floating point") from exc
    if values.ndim != 1 or not np.all(np.isfinite(values)):
        raise PyLCInputError("The time array must contain finite one-dimensional values")
    return values, scalar


def _target(ra: float, dec: float) -> SkyCoord:
    try:
        return SkyCoord(float(ra) * u.deg, float(dec) * u.deg, frame="icrs")
    except (TypeError, ValueError) as exc:
        raise PyLCInputError("The target coordinates are not valid") from exc


def _delay(time: Time, target: SkyCoord, kind: str) -> np.ndarray:
    # The built-in solar-system ephemeris ships with Astropy and does not
    # download JPL kernels.  Geocentric timing matches the original HOPS path.
    return np.asarray(
        time.light_travel_time(
            target,
            kind=kind,
            location=_GEOCENTER,
            ephemeris="builtin",
        ).to_value(u.day),
        dtype=float,
    )


def _observer_time(
    corrected_jd: np.ndarray,
    target: SkyCoord,
    *,
    kind: str,
    scale: str,
) -> Time:
    """Invert a light-travel-time-corrected Julian date."""
    observer = Time(corrected_jd, format="jd", scale=scale)
    for _ in range(5):
        observer = Time(
            corrected_jd - _delay(observer, target, kind),
            format="jd",
            scale=scale,
        )
    return observer


def _result(values: np.ndarray, scalar: bool):
    return float(values[0]) if scalar else np.asarray(values, dtype=float)


def convert_to_bjd_tdb(ra, dec, time_array, time_format):
    """Convert the HOPS-supported time formats to BJD TDB without ExoClock."""
    values, scalar = _values(time_array)
    target = _target(ra, dec)
    time_format = str(time_format).upper()

    if time_format in {"BJD_TDB", "BJD_TT"}:
        converted = values
    elif time_format == "JD_UTC":
        observer = Time(values, format="jd", scale="utc")
        converted = observer.tdb.jd + _delay(observer, target, "barycentric")
    elif time_format == "MJD_UTC":
        observer = Time(values, format="mjd", scale="utc")
        converted = observer.tdb.jd + _delay(observer, target, "barycentric")
    elif time_format == "BJD_UTC":
        converted = Time(values, format="jd", scale="utc").tdb.jd
    elif time_format in {"HJD_TDB", "HJD_TT", "HJD_UTC"}:
        scale = "utc" if time_format == "HJD_UTC" else "tdb"
        observer = _observer_time(values, target, kind="heliocentric", scale=scale)
        converted = observer.tdb.jd + _delay(observer, target, "barycentric")
    else:
        raise PyLCInputError(
            "Not valid time format. Available formats: JD_UTC, MJD_UTC, BJD_UTC, "
            "BJD_TDB, BJD_TT, HJD_UTC, HJD_TDB, HJD_TT"
        )
    return _result(np.asarray(converted, dtype=float), scalar)


def convert_to_jd_utc(ra, dec, time_array, time_format):
    """Convert the HOPS-supported time formats to geocentric JD UTC."""
    values, scalar = _values(time_array)
    target = _target(ra, dec)
    time_format = str(time_format).upper()

    if time_format == "JD_UTC":
        converted = values
    elif time_format == "MJD_UTC":
        converted = values + 2_400_000.5
    elif time_format in {"BJD_TDB", "BJD_TT"}:
        converted = _observer_time(
            values, target, kind="barycentric", scale="tdb"
        ).utc.jd
    elif time_format == "BJD_UTC":
        converted = _observer_time(
            values, target, kind="barycentric", scale="utc"
        ).utc.jd
    elif time_format in {"HJD_TDB", "HJD_TT", "HJD_UTC"}:
        scale = "utc" if time_format == "HJD_UTC" else "tdb"
        converted = _observer_time(
            values, target, kind="heliocentric", scale=scale
        ).utc.jd
    else:
        raise PyLCInputError(
            "Not valid time format. Available formats: JD_UTC, MJD_UTC, BJD_UTC, "
            "BJD_TDB, BJD_TT, HJD_UTC, HJD_TDB, HJD_TT"
        )
    return _result(np.asarray(converted, dtype=float), scalar)
