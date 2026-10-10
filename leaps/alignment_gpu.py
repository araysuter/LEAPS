"""Bounded GPU star catalogs for registration, not photometry.

Only compact (x, y) catalogs survive each batch. Pixel centers use the HOPS
convention (the first pixel is at 0.5, 0.5). GPU libraries are imported lazily
so CPU matching workers do not initialize a GPU runtime.
"""

from __future__ import annotations

import logging
import os
import sys
from typing import Any, Protocol

import numpy as np

# One GPU thread per pixel, across every image in a batch. Most threads exit
# after the threshold test; only isolated peaks compute background-subtracted
# moments. Reject nonfinite/saturated patches and narrow hot pixels.
_CENTROID_BODY = r"""
    int b = i / (h*w), y = (i / w) % h, x = i % w;
    float mean = params[b*4], std = params[b*4+1];
    float psf = params[b*4+2], burn = params[b*4+3];
    int r = max(2, (int)ceil(3*psf));
    int peak_r = max(1, (int)round(psf));
    float peak = pixels[i];
    if (x < r || y < r || x >= w-r || y >= h-r ||
        !isfinite(peak) || peak <= mean+4*std || peak >= burn-1) return;
    int bright = 0;
    for (int dy=-peak_r; dy<=peak_r; ++dy) {
        for (int dx=-peak_r; dx<=peak_r; ++dx) {
            int j = (int)i+dy*w+dx;
            float v = pixels[j];
            if (!isfinite(v) || v > peak || (v == peak && j < (int)i)) return;
            if (v > mean+3*std) ++bright;
        }
    }
    if (2*bright < (2*peak_r+1)*(2*peak_r+1)) return;
    // A local background follows gradients without weighting negative noise
    // upward. The border is outside most of the star's light.
    float sky = 0; int nsky = 0;
    for (int dy=-r; dy<=r; ++dy) {
        for (int dx=-r; dx<=r; ++dx) {
            float v = pixels[(int)i+dy*w+dx];
            if (!isfinite(v) || v >= burn-1) return;
            if (abs(dx) == r || abs(dy) == r) { sky += v; ++nsky; }
        }
    }
    sky /= nsky;
    float sum=0, sx=0, sy=0, sxx=0, syy=0;
    for (int dy=-r; dy<=r; ++dy) {
        for (int dx=-r; dx<=r; ++dx) {
            float v = pixels[(int)i+dy*w+dx]-sky;
            sum += v; sx += v*dx; sy += v*dy;
            sxx += v*dx*dx; syy += v*dy*dy;
        }
    }
    if (sum <= 0) return;
    float cx=sx/sum, cy=sy/sum;
    float vx=sxx/sum-cx*cx, vy=syy/sum-cy*cy;
    float low=0.25*psf*psf, high=2.25*psf*psf;
    if (vx < low || vy < low || vx > high || vy > high ||
        fabs(cx) > psf || fabs(cy) > psf) return;
    // Recenter a Gaussian window to suppress nearby stars and background noise.
    // For a Gaussian source the weighted center converges to the fitted center.
    for (int iteration=0; iteration<8; ++iteration) {
        float weighted=0, wx=0, wy=0;
        for (int dy=-r; dy<=r; ++dy) {
            for (int dx=-r; dx<=r; ++dx) {
                float ex=dx-cx, ey=dy-cy;
                float v=(pixels[(int)i+dy*w+dx]-sky)*exp(-(ex*ex+ey*ey)/(2*psf*psf));
                weighted+=v; wx+=v*dx; wy+=v*dy;
            }
        }
        if (weighted <= 0) return;
        cx=wx/weighted; cy=wy/weighted;
    }
    if (!isfinite(cx) || !isfinite(cy) || fabs(cx)>psf || fabs(cy)>psf) return;
"""

_METAL_SOURCE = (
    "uint i = thread_position_in_grid.x;\n"
    "int h = pixels_shape[1], w = pixels_shape[2];\n"
    "flux[i] = 0; xpos[i] = 0; ypos[i] = 0;\n"
    + _CENTROID_BODY
    + "flux[i] = sum; xpos[i] = x+0.5+cx; ypos[i] = y+0.5+cy;\n"
)

# OpenCL 1.2 is sufficient, including on older NVIDIA/AMD/Intel drivers. An
# atomic append avoids copying full-resolution maps over PCIe to the CPU.
# Capacity overflow is detected and must never silently discard stars.
_OPENCL_SOURCE = (
    """
    __kernel void detect_stars(
        __global const float *pixels, __global const float *params,
        int h, int w, uint capacity,
        volatile __global unsigned int *counts, __global float *catalogs) {
    uint i = get_global_id(0);
    """
    + _CENTROID_BODY
    + """
    uint slot = atomic_inc(counts+b);
    if (slot < capacity) {
        uint offset = (b*capacity+slot)*3;
        catalogs[offset] = x+0.5+cx;
        catalogs[offset+1] = y+0.5+cy;
        catalogs[offset+2] = sum;
    }
    }
    """
)

_LOG = logging.getLogger(__name__)


class StarDetector(Protocol):
    name: str
    backend: str
    batch_pixel_bytes: int
    max_batch_frames: int

    def detect(self, images: list[np.ndarray], headers: list[Any]) -> list[np.ndarray]: ...


def _batch_parameters(images: list[np.ndarray], headers: list[Any]) -> np.ndarray:
    from hops.hops_tools.image_analysis import image_burn_limit, image_mean_std

    if not images or len(images) != len(headers):
        raise ValueError("A header is required for each image")
    if any(image.ndim != 2 or image.shape != images[0].shape or not image.size for image in images):
        raise ValueError("GPU batches require matching nonempty two-dimensional images")
    params = []
    for data, header in zip(images, headers, strict=True):
        mean, std = float(header.get("HOPSMEAN", np.nan)), float(header.get("HOPSSTD", np.nan))
        if not np.isfinite([mean, std]).all() or std <= 0:
            mean, std = image_mean_std(data)
        psf = float(header.get("HOPSPSF", 2))
        if not np.isfinite(psf) or not 0.5 <= psf <= 10:
            raise ValueError("PSF requires the CPU detector")
        params.append([mean, max(std, 1e-6), psf, image_burn_limit(header)])
    return np.asarray(params, dtype=np.float32)


def _select_stars(catalog: np.ndarray, psf: float) -> np.ndarray:
    catalog = catalog[np.isfinite(catalog).all(axis=1) & (catalog[:, 2] > 0)]
    # Break equal-flux ties by position, independent of GPU execution order.
    catalog = catalog[np.lexsort((catalog[:, 1], catalog[:, 0], -catalog[:, 2]))]
    chosen = []
    for star in catalog:
        if not chosen or np.min(np.linalg.norm(np.asarray(chosen) - star[:2], axis=1)) > 2 * psf:
            chosen.append(star[:2])
        if len(chosen) == 60:
            break
    return np.asarray(chosen, dtype=float).reshape(-1, 2)


def _check_detector(detector: StarDetector) -> None:
    """Prove that the driver compiles and executes this kernel correctly."""
    y, x = np.mgrid[:48, :48]
    pixels = 100 + 1000 * np.exp(-((x - 23.2) ** 2 + (y - 22.7) ** 2) / 8)
    header = {"HOPSMEAN": 100, "HOPSSTD": 2, "HOPSPSF": 2, "BITPIX": -32}
    catalogs = detector.detect([pixels], [header])
    if (
        len(catalogs) != 1
        or catalogs[0].shape != (1, 2)
        or not np.allclose(catalogs[0][0], [23.7, 23.2], atol=0.1, rtol=0)
    ):
        raise RuntimeError("GPU star detection failed its startup accuracy check")


class MetalStarDetector:
    backend = "metal"
    name = "Apple GPU (Metal)"
    # Bound all intermediate arrays, not just the input pixels.
    batch_pixel_bytes = 32 * 1024 * 1024
    max_batch_frames = 4

    def __init__(self) -> None:
        import mlx.core as mx

        if not mx.metal.is_available():
            raise RuntimeError("Metal is unavailable")
        self.mx = mx
        self.kernel = mx.fast.metal_kernel(
            name="leaps_alignment_stars",
            input_names=["pixels", "params"],
            output_names=["flux", "xpos", "ypos"],
            source=_METAL_SOURCE,
        )

    def detect(self, images: list[np.ndarray], headers: list[Any]) -> list[np.ndarray]:
        params = _batch_parameters(images, headers)
        mx = self.mx
        # A single stream owns the complete batch; CPU pool workers never use MLX.
        with mx.stream(mx.gpu):
            pixels = mx.array(np.stack(images).astype(np.float32, copy=False))
            outputs = self.kernel(
                inputs=[pixels, mx.array(params, dtype=mx.float32)],
                grid=(pixels.size, 1, 1),
                threadgroup=(min(256, pixels.size), 1, 1),
                output_shapes=[pixels.shape] * 3,
                output_dtypes=[mx.float32] * 3,
            )
            flux, x, y = [value.reshape(len(images), -1) for value in outputs]
            k = min(120, flux.shape[1])
            indices = mx.argpartition(-flux, kth=k - 1, axis=1)[:, :k]
            compact = mx.stack([mx.take_along_axis(a, indices, axis=1) for a in (x, y, flux)], axis=-1)
            mx.eval(compact)
            catalogs = np.array(compact)
        return [_select_stars(catalog, param[2]) for catalog, param in zip(catalogs, params, strict=True)]


class OpenCLStarDetector:
    """Use a vendor GPU driver directly; no CUDA toolkit or SDK is required."""

    backend = "opencl"
    max_batch_frames = 4
    max_candidates = 16384

    def __init__(self, device: Any) -> None:
        import pyopencl as cl

        self.cl = cl
        self.name = f"{device.vendor.strip()} {device.name.strip()} (OpenCL)"
        self.context = cl.Context(devices=[device])
        self.queue = cl.CommandQueue(self.context, device=device)
        self.program = cl.Program(self.context, _OPENCL_SOURCE).build(options=["-cl-std=CL1.2"])
        self.kernel = cl.Kernel(self.program, "detect_stars")
        self.batch_pixel_bytes = max(
            1,
            min(
                32 * 1024 * 1024,
                int(device.max_mem_alloc_size) // 2,
                int(device.global_mem_size) // 16,
            ),
        )
        self.max_allocation = int(device.max_mem_alloc_size)

    def detect(self, images: list[np.ndarray], headers: list[Any]) -> list[np.ndarray]:
        params = _batch_parameters(images, headers)
        pixels = np.stack(images).astype(np.float32, copy=False)
        cl = self.cl
        count = len(images)
        catalogs = np.empty((count, self.max_candidates, 3), dtype=np.float32)
        counts = np.zeros(count, dtype=np.uint32)
        if max(pixels.nbytes, catalogs.nbytes) > self.max_allocation:
            raise ValueError("Image exceeds the GPU's maximum buffer size")
        flags = cl.mem_flags
        pixel_buffer = cl.Buffer(self.context, flags.READ_ONLY | flags.COPY_HOST_PTR, hostbuf=pixels)
        param_buffer = cl.Buffer(self.context, flags.READ_ONLY | flags.COPY_HOST_PTR, hostbuf=params)
        count_buffer = cl.Buffer(self.context, flags.READ_WRITE | flags.COPY_HOST_PTR, hostbuf=counts)
        catalog_buffer = cl.Buffer(self.context, flags.WRITE_ONLY, size=catalogs.nbytes)
        self.kernel.set_args(
            pixel_buffer,
            param_buffer,
            np.int32(pixels.shape[1]),
            np.int32(pixels.shape[2]),
            np.uint32(self.max_candidates),
            count_buffer,
            catalog_buffer,
        )
        cl.enqueue_nd_range_kernel(self.queue, self.kernel, (pixels.size,), None)
        cl.enqueue_copy(self.queue, counts, count_buffer, is_blocking=True)
        if np.any(counts > self.max_candidates):
            raise ValueError("Too many GPU star candidates; retry with the CPU detector")
        cl.enqueue_copy(self.queue, catalogs, catalog_buffer, is_blocking=True)
        return [
            _select_stars(catalog[: int(size)], param[2])
            for catalog, size, param in zip(catalogs, counts, params, strict=True)
        ]


def _opencl_devices(cl: Any) -> list[Any]:
    """Enumerate GPU devices, including integrated graphics, without prompts."""
    devices = []
    for platform in cl.get_platforms():
        try:
            candidates = platform.get_devices(device_type=cl.device_type.GPU)
        except cl.Error:
            continue  # One broken vendor ICD must not hide another vendor's GPU.
        for device in candidates:
            try:
                if device.available and device.compiler_available and device.type & cl.device_type.GPU:
                    devices.append(device)
            except cl.Error:
                continue

    # Prefer discrete devices, then memory capacity. Integrated GPUs remain
    # candidates when a discrete device fails its build/execution probe.
    def rank(device):
        try:
            return (bool(device.host_unified_memory), -int(device.global_mem_size), device.name)
        except cl.Error:
            return (True, 0, "")

    return sorted(devices, key=rank)


def _create_opencl_detector() -> OpenCLStarDetector | None:
    try:
        import pyopencl as cl

        devices = _opencl_devices(cl)
    except Exception as exc:
        _LOG.debug("OpenCL discovery unavailable: %s", exc)
        return None
    for device in devices:
        try:
            detector = OpenCLStarDetector(device)
            _check_detector(detector)
            return detector
        except Exception as exc:
            _LOG.debug("OpenCL device failed its startup probe: %s", exc)
    return None


def create_detector() -> StarDetector | None:
    requested = os.getenv("LEAPS_ALIGNMENT_DEVICE", "auto").lower()
    if requested == "cpu":
        return None
    if requested not in {"auto", "metal", "opencl"}:
        _LOG.warning("Unknown LEAPS_ALIGNMENT_DEVICE %r; using CPU", requested)
        return None
    if sys.platform == "darwin" and requested in {"auto", "metal"}:
        try:
            detector = MetalStarDetector()
            _check_detector(detector)
            return detector
        except Exception as exc:
            _LOG.debug("Metal unavailable: %s", exc)
    if requested in {"auto", "opencl"}:
        return _create_opencl_detector()
    return None


def match_catalogs(reference: np.ndarray, stars: np.ndarray) -> dict[str, Any]:
    """CPU process entry point: match and validate without any image/FITS I/O."""
    from scipy.spatial import cKDTree

    from hops.thirdparty import twirl

    count = min(20, len(reference), len(stars))
    if count < 5:
        raise ValueError("Too few GPU stars; retry with the CPU detector")
    matrix = np.asarray(twirl.utils.find_transform(reference[:count], stars[:count], n=count, tolerance=12))
    if matrix.shape != (3, 3) or not np.isfinite(matrix).all():
        raise ValueError("Invalid star transform")
    scales = np.linalg.svd(matrix[:2, :2], compute_uv=False)
    if np.linalg.det(matrix[:2, :2]) <= 0 or np.any(np.abs(scales - 1) > 0.1):
        raise ValueError("Implausible star transform")
    predicted = reference @ matrix[:2, :2].T + matrix[:2, 2]
    distances, indexes = cKDTree(stars).query(predicted)
    good = distances < 1.0
    matched = len(np.unique(indexes[good]))
    if matched < max(5, count // 2) or np.median(distances[good]) > 0.5:
        raise ValueError("GPU catalog match is uncertain; retry with the CPU detector")
    return {
        "matrix": matrix.tolist(),
        "matched": matched,
        "x0": float(matrix[0, 2]),
        "y0": float(matrix[1, 2]),
        "rotation": float(np.arctan2(matrix[1, 0], matrix[0, 0])),
    }
