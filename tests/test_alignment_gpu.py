from __future__ import annotations

import multiprocessing
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pytest
from scipy.spatial import cKDTree

from leaps.alignment_gpu import MetalStarDetector, OpenCLStarDetector, _opencl_devices, match_catalogs


def _field(shift=(0, 0), angle=0):
    rng = np.random.default_rng(13)
    stars = rng.uniform(40, 470, (25, 2))
    stars = np.array(
        [s for i, s in enumerate(stars) if i == 0 or np.min(np.linalg.norm(stars[:i] - s, axis=1)) > 25]
    )
    rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    stars = stars @ rotation.T + shift
    y, x = np.mgrid[:512, :512]
    data = 100 + 0.01 * x + rng.normal(0, 2, x.shape)
    for i, (sx, sy) in enumerate(stars):
        data += (4000 - i * 90) * np.exp(-((x + 0.5 - sx) ** 2 + (y + 0.5 - sy) ** 2) / 8)
    # Neither a saturated star, a hot pixel, nor an edge source is a usable star.
    data += 90000 * np.exp(-((x - 20) ** 2 + (y - 30) ** 2) / 8)
    data[15, 15] = 30000
    data[0, 200] = 10000
    data[510, 510] = np.nan
    return data.astype(np.float32), stars


def _metal():
    try:
        return MetalStarDetector()
    except (ImportError, RuntimeError) as exc:
        pytest.skip(f"Metal hardware unavailable: {exc}")


def _opencl():
    cl = pytest.importorskip("pyopencl")
    try:
        devices = _opencl_devices(cl)
    except cl.Error as exc:
        pytest.skip(f"OpenCL hardware unavailable: {exc}")
    if not devices:
        pytest.skip("No OpenCL GPU is available")
    # A build/execution failure on available hardware must fail, not skip.
    return OpenCLStarDetector(devices[0])


@pytest.mark.parametrize("backend", ["metal", "opencl"])
def test_gpu_batch_centroids_and_registration_accuracy(backend):
    detector = _metal() if backend == "metal" else _opencl()
    first, stars = _field()
    second, moved = _field((2.25, -1.75), 0.002)
    header = {"HOPSMEAN": 102.5, "HOPSSTD": 2, "HOPSPSF": 2, "BITPIX": -32}
    catalogs = detector.detect([first, second], [header, header])
    for detected, expected in zip(catalogs, (stars, moved), strict=True):
        distances, _ = cKDTree(detected).query(expected)
        assert len(detected) == len(expected)
        assert np.max(distances) < 0.1
    result = match_catalogs(*catalogs)
    assert result["x0"] == pytest.approx(2.25, abs=0.05)
    assert result["y0"] == pytest.approx(-1.75, abs=0.05)
    assert result["rotation"] == pytest.approx(0.002, abs=0.0001)


def test_catalog_matching_runs_in_spawned_cpu_processes():
    _, stars = _field()
    with ProcessPoolExecutor(max_workers=2, mp_context=multiprocessing.get_context("spawn")) as pool:
        futures = [pool.submit(match_catalogs, stars, stars + (i, -i / 2)) for i in (1, 3)]
        results = [future.result(timeout=30) for future in futures]
    assert [r["x0"] for r in results] == pytest.approx([1, 3])
    assert [r["y0"] for r in results] == pytest.approx([-0.5, -1.5])


def test_catalog_validation_rejects_bad_transforms(monkeypatch):
    from hops.thirdparty import twirl

    _, stars = _field()
    for matrix in (
        np.zeros((3, 3)),
        np.full((3, 3), np.nan),
        np.diag([2, 2, 1]),
        np.diag([-1, 1, 1]),
        np.array([[1, 0, 1000], [0, 1, 1000], [0, 0, 1]]),
    ):
        monkeypatch.setattr(twirl.utils, "find_transform", lambda *a, **kw: matrix)
        with pytest.raises(ValueError):
            match_catalogs(stars, stars)
    with pytest.raises(ValueError, match="Too few"):
        match_catalogs(stars[:4], stars)


def test_opencl_matches_metal_catalogs():
    opencl, metal = _opencl(), _metal()
    image, _ = _field()
    header = {"HOPSMEAN": 102.5, "HOPSSTD": 2, "HOPSPSF": 2, "BITPIX": -32}
    cl_catalog = opencl.detect([image], [header])[0]
    metal_catalog = metal.detect([image], [header])[0]
    np.testing.assert_allclose(cl_catalog, metal_catalog, atol=0.002, rtol=0)


def test_opencl_reports_candidate_overflow_instead_of_truncating():
    detector = _opencl()
    detector.max_candidates = 1
    image, _ = _field()
    header = {"HOPSMEAN": 102.5, "HOPSSTD": 2, "HOPSPSF": 2, "BITPIX": -32}
    with pytest.raises(ValueError, match="Too many GPU star candidates"):
        detector.detect([image], [header])


def _fake_opencl(monkeypatch, platforms):
    import sys
    from types import SimpleNamespace

    class DriverError(Exception):
        pass

    cl = SimpleNamespace(
        Error=DriverError, device_type=SimpleNamespace(GPU=4), get_platforms=lambda: platforms
    )
    monkeypatch.setitem(sys.modules, "pyopencl", cl)
    return cl


def _device(name, vendor="NVIDIA", unified=False, available=True, compiler=True, kind=4):
    from types import SimpleNamespace

    return SimpleNamespace(
        name=name,
        vendor=vendor,
        host_unified_memory=unified,
        global_mem_size=2**30,
        type=kind,
        available=available,
        compiler_available=compiler,
    )


def _platform(devices):
    from types import SimpleNamespace

    return SimpleNamespace(get_devices=lambda **kwargs: devices)


def test_opencl_detects_integrated_gpus_and_ignores_cpu_devices(monkeypatch):
    integrated = _device("Iris Xe", "Intel", unified=True)
    discrete = _device("GeForce", "NVIDIA")
    amd = _device("Radeon", "AMD")
    cl = _fake_opencl(
        monkeypatch,
        [
            _platform(
                [
                    integrated,
                    discrete,
                    amd,
                    _device("CPU", kind=2),
                    _device("Unavailable", available=False),
                    _device("No compiler", compiler=False),
                ]
            )
        ],
    )
    assert _opencl_devices(cl) == [discrete, amd, integrated]


def test_opencl_device_failure_tries_remaining_vendors(monkeypatch):
    from leaps import alignment_gpu

    failed = _device("GeForce")
    integrated = _device("Iris Xe", "Intel", unified=True)
    _fake_opencl(monkeypatch, [_platform([failed, integrated])])
    attempted = []

    def factory(device):
        attempted.append(device.name)
        if device is failed:
            raise RuntimeError("Driver could not compile this kernel")
        return device

    monkeypatch.setattr(alignment_gpu, "OpenCLStarDetector", factory)
    checked = []
    monkeypatch.setattr(alignment_gpu, "_check_detector", checked.append)
    assert alignment_gpu._create_opencl_detector() is integrated
    assert attempted == ["GeForce", "Iris Xe"]
    assert checked == [integrated]


def test_opencl_broken_platform_does_not_hide_working_gpu(monkeypatch):
    from types import SimpleNamespace

    device = _device("Radeon", "AMD")
    cl = _fake_opencl(monkeypatch, [])

    def broken(**kwargs):
        raise cl.Error("Broken vendor ICD")

    cl.get_platforms = lambda: [SimpleNamespace(get_devices=broken), _platform([device])]
    assert _opencl_devices(cl) == [device]


def test_detector_routing_and_cpu_fallback(monkeypatch):
    from leaps import alignment_gpu

    metal, opencl = object(), object()
    monkeypatch.setattr(alignment_gpu, "MetalStarDetector", lambda: metal)
    monkeypatch.setattr(alignment_gpu, "_check_detector", lambda detector: None)
    monkeypatch.setattr(alignment_gpu, "_create_opencl_detector", lambda: opencl)
    monkeypatch.setenv("LEAPS_ALIGNMENT_DEVICE", "auto")
    monkeypatch.setattr(alignment_gpu.sys, "platform", "win32")
    assert alignment_gpu.create_detector() is opencl
    monkeypatch.setattr(alignment_gpu.sys, "platform", "darwin")
    assert alignment_gpu.create_detector() is metal
    monkeypatch.setenv("LEAPS_ALIGNMENT_DEVICE", "opencl")
    assert alignment_gpu.create_detector() is opencl
    monkeypatch.setenv("LEAPS_ALIGNMENT_DEVICE", "cpu")
    assert alignment_gpu.create_detector() is None
    monkeypatch.setenv("LEAPS_ALIGNMENT_DEVICE", "auto")
    monkeypatch.setattr(alignment_gpu, "MetalStarDetector", lambda: (_ for _ in ()).throw(RuntimeError()))
    assert alignment_gpu.create_detector() is opencl
    monkeypatch.setattr(alignment_gpu, "_create_opencl_detector", lambda: None)
    assert alignment_gpu.create_detector() is None


def test_no_opencl_driver_is_a_normal_cpu_fallback(monkeypatch):
    from leaps import alignment_gpu

    cl = _fake_opencl(monkeypatch, [])
    assert alignment_gpu._create_opencl_detector() is None

    def unavailable():
        raise cl.Error("PLATFORM_NOT_FOUND_KHR")

    cl.get_platforms = unavailable
    assert alignment_gpu._create_opencl_detector() is None
