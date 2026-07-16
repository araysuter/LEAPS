import tomllib
from pathlib import Path

ROOT = Path(__file__).parents[1]


def test_all_release_inputs_exist_and_are_nonempty() -> None:
    required = (
        ".github/workflows/release.yml",
        "leaps/app.py",
        "leaps/assets/leaps-app-icon.png",
        "leaps/assets/leaps-app-icon.icns",
        "leaps/assets/leaps-app-icon.ico",
        "leaps/assets/leaps-logo-source.png",
        "packaging/LEAPS-macos.spec",
        "packaging/LEAPS-windows.spec",
        "packaging/entitlements.plist",
        "packaging/leaps.iss",
        "scripts/build_macos.sh",
        "scripts/build_windows.ps1",
        "tests/test_windows_regressions.py",
    )

    missing = [path for path in required if not (ROOT / path).is_file()]
    empty = [path for path in required if (ROOT / path).is_file() and not (ROOT / path).stat().st_size]

    assert not missing, f"Missing release inputs: {missing}"
    assert not empty, f"Empty release inputs: {empty}"


def test_requirements_match_project_runtime_dependencies() -> None:
    with (ROOT / "pyproject.toml").open("rb") as handle:
        dependencies = set(tomllib.load(handle)["project"]["dependencies"])
    requirements = {
        line.strip()
        for line in (ROOT / "requirements.txt").read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }

    assert requirements == dependencies


def test_retired_hops_assets_are_archived_without_moving_active_science_code() -> None:
    archive = ROOT / "Hops Archive"
    archived_files = {
        path.relative_to(archive).as_posix()
        for path in archive.rglob("*")
        if path.is_file()
    }

    assert {
        "README.md",
        "logo.jpg",
        "hops/files/close.gif",
        "hops/files/fitting_output_description.txt",
        "hops/files/holomon.gif",
        "hops/files/log.yaml",
        "hops/files/log_profile.yaml",
        "hops/files/log_user.yaml",
        "hops/files/logo.jpg",
        "hops/files/photometry_output_description.txt",
        "hops/files/reduction_output_description.txt",
    } <= archived_files
    assert not (ROOT / "logo.jpg").exists()
    assert not (ROOT / "hops" / "files").exists()
    assert (ROOT / "hops" / "hops_tools" / "image_analysis.py").is_file()
    assert (ROOT / "hops" / "thirdparty" / "twirl" / "utils.py").is_file()
    assert (ROOT / "hops" / "pylightcurve41" / "__init__.py").is_file()
    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert '"files/*_output_description.txt"' not in pyproject


def test_release_scientific_stack_stays_within_validated_versions() -> None:
    with (ROOT / "pyproject.toml").open("rb") as handle:
        dependencies = tomllib.load(handle)["project"]["dependencies"]

    assert "matplotlib>=3.8,<3.11" in dependencies
    assert "numpy>=1.26,<2.5" in dependencies
    assert "astropy>=6.0,<8" in dependencies
    assert "scipy>=1.10,<1.18" in dependencies


def test_deployment_includes_photutils_dynamic_modules() -> None:
    spec = (ROOT / "pysidedeploy.spec").read_text(encoding="utf-8")
    assert "--include-package=photutils" in spec
    assert "--include-package-data=photutils" in spec


def test_macos_bundle_has_stable_privacy_metadata_and_is_resigned() -> None:
    with (ROOT / "pyproject.toml").open("rb") as handle:
        version = tomllib.load(handle)["project"]["version"]
    script = (ROOT / "scripts" / "build_macos.sh").read_text(encoding="utf-8")
    assert "PyInstaller" in script
    assert "packaging/LEAPS-macos.spec" in script
    assert 'CFBundleIdentifier "org.leaps.exoplanet"' in script
    assert "NSDocumentsFolderUsageDescription" in script
    assert "NSRemovableVolumesUsageDescription" in script
    assert 'codesign --force --deep --sign - "$APP"' in script
    assert "--packaging-self-test" in script
    assert 'VERSION="${LEAPS_VERSION:-$PROJECT_VERSION}"' in script
    assert 'CFBundleShortVersionString "$VERSION"' in script

    spec = (ROOT / "packaging" / "LEAPS-macos.spec").read_text(encoding="utf-8")
    assert '"hops", "exotethys", "photutils"' in spec
    assert 'collect_data_files("exoclock")' in spec
    assert 'collect_data_files("astroquery")' in spec
    assert 'collect_data_files("pyvo")' in spec
    assert '"matplotlib.backends.backend_pdf"' in spec
    assert '"leaps-app-icon.icns"' in spec
    assert "NSDocumentsFolderUsageDescription" in spec
    assert "NSRemovableVolumesUsageDescription" in spec
    assert 'target_arch="arm64"' in spec
    assert f'VERSION = os.environ.get("LEAPS_VERSION", "{version}")' in spec


def test_windows_build_uses_fast_pyinstaller_bundle_and_runs_self_test() -> None:
    script = (ROOT / "scripts" / "build_windows.ps1").read_text(encoding="utf-8")
    assert "PyInstaller" in script
    assert "packaging/LEAPS-windows.spec" in script
    assert '"dist/LEAPS"' in script
    assert "--packaging-self-test" in script
    assert "--windows-packaging-self-test" in script
    assert '$env:LEAPS_VERSION = $Version' in script
    assert '"/DSourceDir=$SourceDir"' in script
    assert '"/DMyAppVersion=$Version"' in script

    spec = (ROOT / "packaging" / "LEAPS-windows.spec").read_text(encoding="utf-8")
    assert 'collect_all(package)' in spec
    assert '"hops", "exotethys", "photutils"' in spec
    assert 'collect_data_files("exoclock")' in spec
    assert 'collect_data_files("astroquery")' in spec
    assert 'collect_data_files("pyvo")' in spec
    assert '"matplotlib.backends.backend_pdf"' in spec
    assert '"leaps-app-icon.ico"' in spec
    assert 'ROOT = Path(SPECPATH).parent' in spec

    app = (ROOT / "leaps" / "app.py").read_text(encoding="utf-8")
    assert "_packaging_fitting_self_test" in app
    assert "_packaging_alignment_self_test" in app
    assert "windows_packaging_self_test" in app
    assert "FittingService().run" in app
    assert "AlignmentService().run" in app
    assert "import pyvo.samp as pyvo_samp" in app
    assert 'importlib.metadata.version("leaps-exoplanet")' in app
    assert "installed_version != __version__" in app

    installer = (ROOT / "packaging" / "leaps.iss").read_text(encoding="utf-8")
    assert 'Source: "{#SourceDir}\\*"' in installer
    assert "recursesubdirs createallsubdirs" in installer


def test_release_workflow_can_build_windows_without_rebuilding_macos() -> None:
    workflow = (ROOT / ".github" / "workflows" / "release.yml").read_text(encoding="utf-8")
    assert "target:" in workflow
    assert "inputs.target == 'windows'" in workflow
    assert "inputs.target == 'macos'" in workflow
    assert 'tags: ["v*"]' in workflow
    assert "bash scripts/build_macos.sh" in workflow
    assert "./scripts/build_windows.ps1" in workflow
    assert "tests/test_windows_regressions.py" in workflow
    assert "Run Windows alignment and UI regressions" in workflow
    assert "artifacts/LEAPS-Windows-x64-Setup.exe" in workflow
    assert "artifacts/LEAPS-Apple-Silicon.dmg" in workflow
    assert "github.ref_type == 'tag'" in workflow
    assert "TAG_VERSION" in workflow


def test_offline_asset_manifest_uses_current_repository_owner() -> None:
    source = (ROOT / "leaps" / "offline.py").read_text(encoding="utf-8")
    assert "https://github.com/araysuter/LEAPS/releases/latest/download/offline-assets.json" in source
    assert "github.com/MrRayBob/LEAPS" not in source


def test_installer_version_matches_release() -> None:
    with (ROOT / "pyproject.toml").open("rb") as handle:
        version = tomllib.load(handle)["project"]["version"]
    installer = (ROOT / "packaging" / "leaps.iss").read_text(encoding="utf-8")
    assert "#ifndef MyAppVersion" in installer
    assert "AppVersion={#MyAppVersion}" in installer
    assert f'#define MyAppVersion "{version}"' in installer
    assert f'__version__ = "{version}"' in (ROOT / "leaps" / "__init__.py").read_text(
        encoding="utf-8"
    )
