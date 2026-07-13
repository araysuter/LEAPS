from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

try:
    import truststore

    truststore.inject_into_ssl()
except ImportError:
    pass

from PySide6.QtCore import QSettings, Qt, QTimer
from PySide6.QtGui import QFont, QIcon
from PySide6.QtWidgets import QApplication

from leaps import __version__
from leaps.ui.main_window import MainWindow
from leaps.ui.settings_dialog import FirstRunDialog
from leaps.ui.theme import APP_STYLESHEET, palette


def packaging_self_test() -> int:
    """Import critical modules and exercise packaged figure export."""
    import emcee
    import h5py
    import numpy
    import photutils.geometry.core as photutils_geometry
    import quantities
    import requests
    import scipy.ndimage
    import scipy.optimize
    import yaml
    from astropy.io import fits
    from astropy.wcs import WCS
    from astroquery.gaia import Gaia
    from matplotlib.backends import backend_agg, backend_pdf
    from matplotlib.figure import Figure
    from photutils.aperture import CircularAperture, aperture_photometry
    from PIL import Image

    from hops.hops_tools import image_analysis

    required = (
        emcee,
        h5py,
        numpy,
        photutils_geometry,
        quantities,
        requests,
        scipy.ndimage,
        scipy.optimize,
        yaml,
        fits,
        WCS,
        Gaia,
        image_analysis,
        backend_agg,
        backend_pdf,
        CircularAperture,
        aperture_photometry,
        Image,
    )
    if any(item is None for item in required):
        return 1
    with tempfile.TemporaryDirectory(prefix="leaps-packaging-test-") as directory:
        figure = Figure(figsize=(2, 2))
        axis = figure.add_subplot(111)
        axis.plot((0, 1), (0, 1))
        png_path = Path(directory) / "figure.png"
        pdf_path = Path(directory) / "figure.pdf"
        figure.savefig(png_path)
        figure.savefig(pdf_path)
        if not png_path.is_file() or not pdf_path.is_file():
            return 1
    print("LEAPS packaged runtime self-test passed", flush=True)
    return 0


def create_application(argv: list[str] | None = None) -> QApplication:
    QApplication.setOrganizationName("LEAPS")
    QApplication.setOrganizationDomain("leaps-astronomy.org")
    QApplication.setApplicationName("LEAPS")
    QApplication.setApplicationVersion(__version__)
    QApplication.setAttribute(Qt.ApplicationAttribute.AA_DontUseNativeMenuBar, False)
    app = QApplication.instance() or QApplication(argv or sys.argv)
    if sys.platform == "darwin":
        app.setFont(QFont(".AppleSystemUIFont"))
    elif sys.platform.startswith("win"):
        app.setFont(QFont("Segoe UI"))
    app.setPalette(palette())
    app.setStyleSheet(APP_STYLESHEET)
    app.setWindowIcon(QIcon(str(Path(__file__).resolve().parent / "assets" / "leaps-app-icon.png")))
    return app


def main() -> int:
    if "--packaging-self-test" in sys.argv:
        return packaging_self_test()
    app = create_application()
    demo = os.getenv("LEAPS_DEMO") == "1"
    window = MainWindow(demo=demo)
    window.show()
    if not demo and os.getenv("LEAPS_SKIP_ONBOARDING") != "1":
        settings = QSettings()
        if not settings.value("setup/complete", False, type=bool):
            setup = FirstRunDialog(window)
            if setup.exec():
                settings.setValue("setup/complete", True)
                if setup.offline_choice.currentIndex() == 1:
                    QTimer.singleShot(0, window.open_settings)
    screenshot = os.getenv("LEAPS_SCREENSHOT_PATH")
    if screenshot:

        def capture() -> None:
            path = Path(screenshot).expanduser().resolve()
            path.parent.mkdir(parents=True, exist_ok=True)
            window.grab().save(str(path))
            app.quit()

        QTimer.singleShot(1000, capture)
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
