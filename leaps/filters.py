from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Passband:
    label: str
    identifier: str
    aliases: tuple[str, ...]


# These are the passbands exposed by the original HOPS Data & Target and
# Fitting windows.  The identifier is the exact PyLightcurve/ExoTETHyS name.
HOPS_PASSBANDS: tuple[Passband, ...] = (
    Passband("Clear", "clear", ("none",)),
    Passband("Luminance", "luminance", ("lum", "l")),
    Passband("Johnson U", "JOHNSON_U", ("u", "uj", "johnsonu")),
    Passband("Johnson B", "JOHNSON_B", ("b", "bj", "johnsonb")),
    Passband("Johnson V", "JOHNSON_V", ("v", "vj", "johnsonv")),
    Passband("Cousins R", "COUSINS_R", ("r", "rc", "cousinsr")),
    Passband("Cousins I", "COUSINS_I", ("i", "ic", "cousinsi")),
    Passband("2MASS H", "2mass_h", ("h", "2massh")),
    Passband("2MASS J", "2mass_j", ("j", "2massj")),
    Passband("2MASS Ks", "2mass_ks", ("k", "ks", "2massk", "2massks")),
    Passband(
        "Astrodon ExoPlanet-BB",
        "exoplanets_bb",
        ("exoplanets", "exoplanetbb", "astrodonexoplanetbb"),
    ),
    Passband("SDSS u'", "sdss_u", ("up", "uprime", "sdssu")),
    Passband("SDSS g'", "sdss_g", ("gp", "gprime", "sdssg")),
    Passband("SDSS r'", "sdss_r", ("rp", "rprime", "sdssr")),
    Passband("SDSS i'", "sdss_i", ("ip", "iprime", "sdssi")),
    Passband("SDSS z'", "sdss_z", ("zp", "zprime", "sdssz")),
    Passband("TESS", "TESS", ("tess band", "tesspassband")),
)


# Display order and labels from the original HOPS observation-filter menu.
# Keep this separate from ``HOPS_PASSBANDS`` because reports use the longer,
# descriptive labels while Data & Target intentionally mirrors HOPS exactly.
HOPS_FILTER_MENU: tuple[tuple[str, str], ...] = (
    ("Clear", "clear"),
    ("Luminance", "luminance"),
    ("U", "JOHNSON_U"),
    ("B", "JOHNSON_B"),
    ("V", "JOHNSON_V"),
    ("R", "COUSINS_R"),
    ("I", "COUSINS_I"),
    ("H", "2mass_h"),
    ("J", "2mass_j"),
    ("K", "2mass_ks"),
    ("Astrodon ExoPlanet-BB", "exoplanets_bb"),
    ("u'", "sdss_u"),
    ("g'", "sdss_g"),
    ("r'", "sdss_r"),
    ("z'", "sdss_z"),
    ("i'", "sdss_i"),
)


def _key(value: object) -> str:
    return "".join(character for character in str(value).casefold() if character.isalnum())


_BY_ALIAS: dict[str, Passband] = {}
for _passband in HOPS_PASSBANDS:
    for _alias in (_passband.label, _passband.identifier, *_passband.aliases):
        _BY_ALIAS[_key(_alias)] = _passband


def normalize_filter(value: object) -> str | None:
    """Return the canonical PyLightcurve passband used by HOPS."""
    if value is None or not str(value).strip():
        return None
    passband = _BY_ALIAS.get(_key(value))
    return passband.identifier if passband else None


def passband_label(identifier: object) -> str:
    passband = _BY_ALIAS.get(_key(identifier))
    return passband.label if passband else str(identifier)


def passband_choices() -> tuple[tuple[str, str], ...]:
    return tuple((passband.label, passband.identifier) for passband in HOPS_PASSBANDS)


def hops_filter_choices() -> tuple[tuple[str, str], ...]:
    """Return the original HOPS filter-menu labels and canonical identifiers."""
    return HOPS_FILTER_MENU
