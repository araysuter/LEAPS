from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .fits_inventory import validate_coordinates
from .models import LEAPSError, StageID


@dataclass(slots=True)
class PlanetParameters:
    name: str
    ra: str
    dec: str
    period: float
    mid_time: float
    rp_over_rs: float
    sma_over_rs: float
    inclination: float
    eccentricity: float
    periastron: float
    metallicity: float
    temperature: float
    logg: float
    source: str
    source_date: str = ""


class PlanetCatalogResolver:
    def __init__(self, nasa_snapshot: str | Path | None = None) -> None:
        self.nasa_snapshot = Path(nasa_snapshot).expanduser() if nasa_snapshot else None

    def resolve(self, ra: str, dec: str, name: str = "") -> PlanetParameters:
        ra, dec = validate_coordinates(ra, dec)
        exoclock_result = self._from_exoclock(ra, dec)
        if exoclock_result:
            return exoclock_result
        nasa_result = self._from_nasa(name, ra, dec)
        if nasa_result:
            return nasa_result
        raise LEAPSError(
            "PLANET_NOT_FOUND",
            "No planet parameters were found",
            "The target coordinates are valid, but neither ExoClock nor the offline NASA snapshot contains a match.",
            ["Enter the planet parameters manually", "Update offline data and retry"],
            stage=StageID.FITTING,
        )

    @staticmethod
    def _from_exoclock(ra: str, dec: str) -> PlanetParameters | None:
        try:
            import exoclock

            data = exoclock.locate_planet(exoclock.Hours(ra), exoclock.Degrees(dec))
            return PlanetParameters(
                name=data["name"],
                ra=str(data["star"]["ra"]),
                dec=str(data["star"]["dec"]),
                period=float(data["planet"]["ephem_period"]),
                mid_time=float(data["planet"]["ephem_mid_time"]),
                rp_over_rs=float(data["planet"]["rp_over_rs"]),
                sma_over_rs=float(data["planet"]["sma_over_rs"]),
                inclination=float(data["planet"]["inclination"]),
                eccentricity=float(data["planet"]["eccentricity"]),
                periastron=float(data["planet"]["periastron"]),
                metallicity=float(data["planet"]["meta"]),
                temperature=float(data["planet"]["teff"]),
                logg=float(data["planet"]["logg"]),
                source="ExoClock",
            )
        except Exception:
            return None

    def _from_nasa(self, name: str, ra: str, dec: str) -> PlanetParameters | None:
        if not self.nasa_snapshot or not self.nasa_snapshot.exists():
            return None
        payload: dict[str, Any] | list[dict[str, Any]] = json.loads(
            self.nasa_snapshot.read_text(encoding="utf-8")
        )
        candidates = payload.get("planets", []) if isinstance(payload, dict) else payload
        normalized = name.casefold().replace(" ", "").replace("-", "")
        for record in candidates:
            record_name = str(record.get("pl_name", ""))
            comparable = record_name.casefold().replace(" ", "").replace("-", "")
            if normalized and comparable != normalized:
                continue
            try:
                return PlanetParameters(
                    name=record_name,
                    ra=ra,
                    dec=dec,
                    period=float(record["pl_orbper"]),
                    mid_time=float(record["pl_tranmid"]),
                    rp_over_rs=float(record["pl_ratror"]),
                    sma_over_rs=float(record["pl_ratdor"]),
                    inclination=float(record["pl_orbincl"]),
                    eccentricity=float(record.get("pl_orbeccen") or 0),
                    periastron=float(record.get("pl_orblper") or 0),
                    metallicity=float(record.get("st_met") or 0),
                    temperature=float(record["st_teff"]),
                    logg=float(record["st_logg"]),
                    source="NASA Exoplanet Archive",
                    source_date=str(payload.get("snapshot_date", "")) if isinstance(payload, dict) else "",
                )
            except (KeyError, TypeError, ValueError):
                continue
        return None
