"""Compatibility entry point for installations upgraded from HOPS."""

from leaps.app import main


def run_app() -> int:
    """Launch the LEAPS desktop application."""
    return main()


if __name__ == "__main__":
    raise SystemExit(run_app())
