from pathlib import Path

ROOT = Path(__file__).parents[1]


def test_deployment_includes_photutils_dynamic_modules() -> None:
    spec = (ROOT / "pysidedeploy.spec").read_text(encoding="utf-8")
    assert "--include-package=photutils" in spec
    assert "--include-package-data=photutils" in spec


def test_macos_bundle_has_stable_privacy_metadata_and_is_resigned() -> None:
    script = (ROOT / "scripts" / "build_macos.sh").read_text(encoding="utf-8")
    assert "PyInstaller" in script
    assert "packaging/LEAPS-macos.spec" in script
    assert 'CFBundleIdentifier "org.leaps.exoplanet"' in script
    assert "NSDocumentsFolderUsageDescription" in script
    assert "NSRemovableVolumesUsageDescription" in script
    assert 'codesign --force --deep --sign - "$APP"' in script
    assert "--packaging-self-test" in script

    spec = (ROOT / "packaging" / "LEAPS-macos.spec").read_text(encoding="utf-8")
    assert '"hops", "exoclock", "exotethys", "photutils"' in spec
    assert 'collect_data_files("astroquery")' in spec
    assert '"matplotlib.backends.backend_pdf"' in spec
    assert 'target_arch="arm64"' in spec


def test_windows_build_uses_fast_pyinstaller_bundle_and_runs_self_test() -> None:
    script = (ROOT / "scripts" / "build_windows.ps1").read_text(encoding="utf-8")
    assert "PyInstaller" in script
    assert "packaging/LEAPS-windows.spec" in script
    assert '"dist/LEAPS"' in script
    assert "--packaging-self-test" in script

    spec = (ROOT / "packaging" / "LEAPS-windows.spec").read_text(encoding="utf-8")
    assert 'collect_all(package)' in spec
    assert '"hops", "exoclock", "exotethys", "photutils"' in spec
    assert 'collect_data_files("astroquery")' in spec
    assert '"matplotlib.backends.backend_pdf"' in spec
    assert 'ROOT = Path(SPECPATH).parent' in spec


def test_release_workflow_can_build_windows_without_rebuilding_macos() -> None:
    workflow = (ROOT / ".github" / "workflows" / "release.yml").read_text(encoding="utf-8")
    assert "target:" in workflow
    assert "inputs.target == 'windows'" in workflow


def test_installer_version_matches_release() -> None:
    installer = (ROOT / "packaging" / "leaps.iss").read_text(encoding="utf-8")
    assert '#define MyAppVersion "1.0.0"' in installer
