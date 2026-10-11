from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
from astropy.io import fits
from astropy.table import Table
from astropy.wcs import WCS

from leaps.models import LEAPSError
from leaps.science import CancellationToken, PlateSolveService
from leaps.targets import ResolvedTarget, TargetNameResolver


def catalogue_solution():
    wcs = WCS(naxis=2)
    wcs.wcs.ctype = ['RA---TAN', 'DEC--TAN']
    wcs.wcs.crpix = [50.0, 50.0]
    wcs.wcs.crval = [100.0, 20.0]
    wcs.wcs.cdelt = [-0.6 / 3600, 0.6 / 3600]
    pixels = np.array([(x, y) for y in (10, 49, 85) for x in (10, 49, 85)], dtype=float)
    world = wcs.all_pix2world(pixels, 0)
    matches = {i: dict(x=p[0], y=p[1], ra=w[0], dec=w[1]) for i, (p, w) in enumerate(zip(pixels, world))}
    return {'plate_solution': wcs, 'identified_stars': matches}, pixels.tolist()


def test_catalogue_validation_rejects_false_matches_and_duplicates():
    solution, stars = catalogue_solution()
    PlateSolveService._validate_catalogue_solution(solution, stars, (100, 100), 2)
    matches = solution['identified_stars']
    solution['identified_stars'] = dict(list(matches.items())[:5])
    with pytest.raises(ValueError, match='Only 5'):
        PlateSolveService._validate_catalogue_solution(solution, stars, (100, 100), 2)
    solution['identified_stars'] = matches
    solution['plate_solution'].wcs.crpix += 6
    with pytest.raises(ValueError, match='residuals'):
        PlateSolveService._validate_catalogue_solution(solution, stars, (100, 100), 2)
    solution['identified_stars'] = {i: matches[0] for i in range(9)}
    with pytest.raises(ValueError, match='distinct'):
        PlateSolveService._validate_catalogue_solution(solution, stars, (100, 100), 2)


def test_plate_solve_recovers_header_target_and_reuses_sparse_cache(tmp_path, monkeypatch):
    import hops.hops_tools.centroids_and_stars as catalogue
    import hops.hops_tools.image_analysis as analysis

    solution, stars = catalogue_solution()
    frame = tmp_path / 'outputs/reduction/frame.fit'
    frame.parent.mkdir(parents=True)
    header = fits.Header({'OBJECT': 'ESPC WASP-12', 'OBJCTRA': '06 40 01', 'OBJCTDEC': '+20 00 00',
                          'HOPSMEAN': 100, 'HOPSSTD': 2, 'HOPSPSF': 2})
    fits.writeto(frame, np.full((100, 100), 100, dtype=np.float32), header)
    original = frame.read_bytes()
    monkeypatch.setattr(analysis, 'image_find_stars', lambda *a, **kw: stars)
    queries, solve_args = [], []
    table = Table({'ra': [m['ra'] for m in solution['identified_stars'].values()],
                   'dec': [m['dec'] for m in solution['identified_stars'].values()]})
    monkeypatch.setattr(catalogue, '_get_gaia_stars', lambda *a, **kw: queries.append(a) or table)
    def solve(*args, **kwargs):
        solve_args.append((args, kwargs))
        return solution
    monkeypatch.setattr(analysis, 'image_plate_solve', solve)
    resolver = TargetNameResolver()
    target = ResolvedTarget('WASP-12', '06:40:00', '+20:00:00', 'test catalogue')
    monkeypatch.setattr(resolver, 'resolve', lambda name: target)
    for _ in range(2):
        result = PlateSolveService().solve(frame, '06:40:01', '+20:00:00', 1.0,
                                           target_name='ESPC WASP-12', target_resolver=resolver)
        assert result.solved
        assert result.resolved_target == target
        assert result.target_xy == pytest.approx((49, 49))
        assert result.attempts[-1].pixel_scale == pytest.approx(.6)
    assert len(queries) == 1  # A valid sparse field is not downloaded on every retry.
    assert solve_args[0][0][2] == pytest.approx(100 + 1 / 240)  # Keep the field pointing hint.
    assert solve_args[0][1]['sip_degree'] is None
    assert frame.read_bytes() == original


def test_cancelled_plate_solve_never_resolves_target(tmp_path, monkeypatch):
    from leaps import science
    calls = []
    monkeypatch.setattr(science, '_read_fits_image', lambda frame: (np.zeros((8, 8)), {}))
    resolver = SimpleNamespace(resolve_header_target=lambda *args: calls.append(args))
    token = CancellationToken()
    token.cancel()
    with pytest.raises(LEAPSError) as failure:
        PlateSolveService().solve(tmp_path / 'frame.fit', '06:40:00', '+20:00:00', 1,
                                   token=token, target_name='WASP-12', target_resolver=resolver)
    assert failure.value.code == 'JOB_CANCELLED'
    assert not calls


def test_catalogue_validation_rejects_a_tiny_matched_region():
    solution, stars = catalogue_solution()
    with pytest.raises(ValueError, match='cover enough'):
        PlateSolveService._validate_catalogue_solution(solution, stars, (1000, 1000), 2)


def test_coordinate_recovery_updates_ui_and_invalidates_downstream_only(tmp_path, qapp):
    from leaps.models import StageID, StageStatus
    from leaps.project import ProjectWorkspace
    from leaps.science import PlateSolveAttempt, PlateSolveResult
    from leaps.ui.main_window import MainWindow

    window = MainWindow(demo=True)
    project = ProjectWorkspace.create(tmp_path, 'Recovery test')
    window.project = project
    project.manifest.target_name = 'ESPC WASP-12'
    project.manifest.target_ra = '06:30:54.29'
    project.manifest.target_dec = '+29:40:13.18'
    for stage in (StageID.ALIGNMENT, StageID.LIGHT_CURVE, StageID.FITTING):
        project.manifest.stages[stage.value].status = StageStatus.COMPLETE
    target = ResolvedTarget('WASP-12', '06:30:32.7966', '+29:40:20.266', 'test catalogue')
    result = PlateSolveResult(True, [PlateSolveAttempt(1, .574, 'complete', '38 matches')],
                              (336, 596), 38, resolved_target=target)
    window._plate_complete(result)
    assert project.manifest.target_ra == target.ra
    assert target.ra in window.plate_page.inspector.coordinates.text()
    assert 'corrected' in window.plate_page.inspector.explanation.text()
    assert project.manifest.stages[StageID.ALIGNMENT.value].status == StageStatus.COMPLETE
    assert project.manifest.stages[StageID.FITTING.value].status == StageStatus.LOCKED
    assert project.manifest.stages[StageID.LIGHT_CURVE.value].status == StageStatus.LOCKED
    saved = ProjectWorkspace.open(tmp_path)
    assert saved.manifest.target_ra == target.ra
    assert saved.manifest.settings['plate_solution']['target_xy'] == [336, 596]
    window.close()


def test_gaia_small_query_uses_http_timeouts_and_parses_catalogue(monkeypatch):
    from io import BytesIO

    import requests

    from hops.hops_tools.centroids_and_stars import _get_gaia_stars

    payload = BytesIO()
    Table({'source_id': [1], 'ra': [100.0], 'dec': [20.0]}).write(payload, format='votable')
    calls = []
    class Response:
        content = payload.getvalue()
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass
        def raise_for_status(self):
            pass
    def post(url, **kwargs):
        calls.append((url, kwargs))
        return Response()
    monkeypatch.setattr(requests, 'post', post)
    table = _get_gaia_stars(100, 20, .5, limit=1000)
    assert list(table['source_id']) == [1]
    assert calls[0][0].endswith('/tap/sync')
    assert calls[0][1]['timeout'] == (5, 20)
    assert 'TOP 1000' in calls[0][1]['data']['QUERY']
    def offline(*args, **kwargs):
        raise requests.Timeout('Gaia unavailable')
    monkeypatch.setattr(requests, 'post', offline)
    with pytest.raises(requests.Timeout):
        _get_gaia_stars(100, 20, .5)
