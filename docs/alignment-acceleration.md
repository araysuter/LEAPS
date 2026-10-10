# Alignment acceleration

Alignment automatically uses MLX/Metal on Apple Silicon or OpenCL through an
available GPU driver on Windows to detect stars in bounded batches before
matching frames on the CPU. Detection retains only up
to 60 star coordinates per frame. A batch contains at most four images and
normally at most 32 MiB of float32 pixels (one oversized image is processed on
its own). Full images are released after each batch, not retained for the run.

The shared Metal/OpenCL kernel finds isolated peaks, rejects saturated/nonfinite patches and
hot pixels, and computes subpixel positions using iteratively recentered
Gaussian-weighted moments. This is a registration detector; the photometry
measurement algorithms are unchanged. Coordinates retain the HOPS pixel-center
convention, with the first pixel centered at (0.5, 0.5).

After detection, a spawned CPU process pool matches the small catalogs with the
existing Twirl algorithm. The pool can use every logical CPU, without the former
four-thread limit or Python's shared interpreter lock. The parent writes FITS
headers and emits progress. Neither images nor GPU objects cross process
boundaries. Only one task per worker is queued, and cancellation cancels queued
work, waits for active work to finish, and prevents committing a partial summary.

Transforms must be finite, preserve orientation, have scale factors within 10%
of unity, and match at least five distinct stars (and half the initial matching
sample) within one pixel, with median residual at most half a pixel. Sparse or
uncertain catalogs retry the established CPU detector. Failure to initialize
any GPU backend, or failure to detect the reference, uses the existing CPU
pipeline for the run. A later GPU error preserves completed catalogs and retries the rest on
CPU. Set `LEAPS_ALIGNMENT_DEVICE=cpu` to select the original pipeline explicitly.

## GPU discovery and Windows support

The default `LEAPS_ALIGNMENT_DEVICE=auto` selects:

1. Metal on Apple Silicon, if its startup accuracy check succeeds.
2. OpenCL GPU devices provided by installed drivers. Discrete GPUs are preferred
   over integrated GPUs, with larger device memory used as a secondary preference.
   Each candidate must compile and execute a small star-detection accuracy check.
   A failed device or vendor platform does not hide other usable GPUs.
3. The established CPU pipeline if no tested GPU is usable.

OpenCL supports NVIDIA, AMD and Intel GPUs, including integrated graphics, **when
an installed driver exposes OpenCL 1.2 or newer with kernel compilation**. Merely
having a display adapter does not establish compute support. OpenCL CPU devices
are excluded so the UI never labels a CPU implementation as GPU acceleration.
The application neither installs nor updates system drivers.

NVIDIA's [OpenCL implementation](https://developer.nvidia.com/opencl) ships with
its graphics driver. This backend does not require CUDA, `nvcc`, or an OpenCL
SDK. The Windows application bundles PyOpenCL and its OpenCL loader DLLs; the
loader discovers vendor drivers registered on the user's machine. See the
[PyOpenCL installation documentation](https://documen.tician.de/pyopencl/misc.html).
CUDA toolkit discovery is unnecessary for this backend, and no CUDA-specific
execution path is included.

For troubleshooting, `LEAPS_ALIGNMENT_DEVICE=opencl` or `metal` requests that
backend (falling back to CPU if unavailable). `cpu` disables GPU probing. Detection
progress names the selected GPU, and records from accelerated runs report
`metal`, `opencl`, or `cpu` for the detector used. OpenCL uses an atomic candidate buffer and transfers
only candidate records to the CPU, rather than full-resolution maps. Candidate
overflow causes a CPU retry rather than silent truncation. Batch size accounts
for the selected device's memory limits.

The Windows dependency is platform-scoped. Linux developers can also install
`pyopencl>=2025.2,<2027` and use this OpenCL path with an appropriate driver;
Linux distribution packaging is not provided here.

## Local validation, October 10, 2026

On an Apple M5 with 32 GB RAM, using reduced WASP-12 images (1375 × 1100 pixels):

| Measurement | Existing CPU path | New Metal/process path |
| --- | ---: | ---: |
| Detection only, 12 frames | 5.48 s | 0.27 s |
| Entire Alignment, 40 frames | 130.87 s | 9.06 s |
| Successful frames in full comparison | 40/40 | 40/40 |

The 40 frames were evenly sampled from the sequence and copied into temporary
projects; original FITS files and the user's project were not modified. The GPU
run used no CPU detection fallbacks. Timings include an independently running
LEAPS alignment on the same Mac, so these are illustrative measurements rather
than an idle-machine performance guarantee. The full comparison includes FITS
reads/writes, GPU detection, process startup, CPU matching, and result commit.

All 11 relative transforms in the 12-frame accuracy sample passed validation.
The greatest disagreement from the existing method, evaluated at the CPU
reference star positions, was 0.272 pixels. This measures agreement with the
existing method, not absolute astrometric truth or final light-curve equivalence.
Synthetic tests additionally cover known translation and rotation, gradients,
noise, saturation, hot pixels, edge sources, and nonfinite pixels, with centroid
error under 0.1 pixel.

## Checks

Run `pytest tests/test_alignment_gpu.py` with GPU access for actual kernel
accuracy and spawned-process matching tests. Each hardware test skips if its
backend/device is absent, but a kernel failure on an available OpenCL GPU fails
the test. The Metal/OpenCL agreement test runs when both are available; orchestration/fallback tests in
`tests/test_science_services.py` also run on CPU-only CI. The ordinary regression
suite explicitly selects CPU detection, keeping its mocks hardware-independent.

Apple Silicon dependencies include MLX, and the macOS PyInstaller spec collects
its libraries and Metal resources. Windows dependencies include PyOpenCL; its
PyInstaller spec collects the package plus `pyopencl.libs` DLLs and load-order
metadata. The packaged `--packaging-self-test` verifies PyOpenCL can import on
Windows even without a GPU, exercises GPU detection when available, and always
exercises a spawned matching process. The `LEAPS_ALIGNMENT_DEVICE=cpu` override
also applies to that self-test.

The OpenCL kernel and Metal/OpenCL numerical agreement have been tested on the
Apple M5 OpenCL driver. A full OpenCL run on temporary copies of eight sampled
WASP-12 frames completed in 3.59 seconds, with eight successful alignments and
no CPU detection retries. The 57 targeted alignment, GPU, Windows-regression,
and packaging tests passed. Unit tests cover NVIDIA/AMD/Intel device selection,
integrated-GPU fallback, absent/broken drivers, candidate overflow, and backend
reporting. Native Windows NVIDIA/AMD/Intel execution and the built Windows
installer still require hardware validation; the Mac timings above are not
Windows performance claims.
