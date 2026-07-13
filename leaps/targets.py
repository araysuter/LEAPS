from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .models import LEAPSError, StageID


@dataclass(slots=True)
class ResolvedTarget:
    name: str
    ra: str
    dec: str
    source: str


class TargetNameResolver:
    """Resolve a target name without making the UI depend on network availability."""

    def __init__(
        self,
        *,
        cache_path: str | Path | None = None,
        nasa_snapshot: str | Path | None = None,
    ) -> None:
        self.cache_path = Path(cache_path).expanduser() if cache_path else None
        self.nasa_snapshot = Path(nasa_snapshot).expanduser() if nasa_snapshot else None

    def resolve(self, name: str) -> ResolvedTarget:
        requested = name.strip()
        if not requested:
            raise LEAPSError(
                "TARGET_NAME_REQUIRED",
                "Enter a target name",
                "Type a star or planet name before looking up coordinates.",
                ["Enter a target name", "Enter coordinates manually"],
                stage=StageID.DATA_TARGET,
            )

        cached = self._from_cache(requested)
        if cached:
            return cached

        offline = self._from_nasa(requested)
        if offline:
            self._remember(requested, offline)
            return offline

        online = self._from_simbad(requested)
        if online:
            self._remember(requested, online)
            return online

        raise LEAPSError(
            "TARGET_NAME_NOT_FOUND",
            "Target name was not found",
            f"No coordinates found for “{requested}”. Check the spelling or enter RA/DEC manually.",
            ["Check the target name", "Enter coordinates manually", "Retry when online"],
            stage=StageID.DATA_TARGET,
        )

    @staticmethod
    def _key(name: str) -> str:
        return "".join(character for character in name.casefold() if character.isalnum())

    def _from_cache(self, name: str) -> ResolvedTarget | None:
        if not self.cache_path or not self.cache_path.exists():
            return None
        try:
            payload = json.loads(self.cache_path.read_text(encoding="utf-8"))
            record = payload.get(self._key(name))
            return ResolvedTarget(**record) if record else None
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            return None

    def _remember(self, requested: str, target: ResolvedTarget) -> None:
        if not self.cache_path:
            return
        try:
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            payload: dict[str, Any] = {}
            if self.cache_path.exists():
                payload = json.loads(self.cache_path.read_text(encoding="utf-8"))
            record = asdict(target)
            payload[self._key(requested)] = record
            payload[self._key(target.name)] = record
            temporary = self.cache_path.with_suffix(self.cache_path.suffix + ".tmp")
            temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
            os.replace(temporary, self.cache_path)
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            pass

    def _from_nasa(self, name: str) -> ResolvedTarget | None:
        if not self.nasa_snapshot or not self.nasa_snapshot.exists():
            return None
        try:
            import astropy.units as units
            from astropy.coordinates import SkyCoord

            payload = json.loads(self.nasa_snapshot.read_text(encoding="utf-8"))
            records = payload.get("planets", []) if isinstance(payload, dict) else payload
            requested = self._key(name)
            for record in records:
                candidates = (record.get("pl_name"), record.get("hostname"), record.get("star_name"))
                if requested not in {self._key(str(candidate)) for candidate in candidates if candidate}:
                    continue
                coordinate = SkyCoord(float(record["ra"]) * units.deg, float(record["dec"]) * units.deg)
                return ResolvedTarget(
                    name=str(record.get("pl_name") or record.get("hostname") or name),
                    ra=coordinate.ra.to_string(unit=units.hourangle, sep=":", precision=2, pad=True),
                    dec=coordinate.dec.to_string(
                        unit=units.deg, sep=":", precision=2, alwayssign=True, pad=True
                    ),
                    source="NASA Exoplanet Archive (offline)",
                )
        except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError):
            return None
        return None

    @staticmethod
    def _from_simbad(name: str) -> ResolvedTarget | None:
        try:
            import exoclock

            target = exoclock.simbad_search_by_name(name, max_trials=1)
            if not target:
                return None
            ra, dec = target.coord().split(maxsplit=1)
            return ResolvedTarget(
                name=str(target.name or name),
                ra=ra,
                dec=dec,
                source="SIMBAD via ExoClock",
            )
        except Exception:
            return None
