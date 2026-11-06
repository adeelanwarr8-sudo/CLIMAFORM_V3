"""ClimaForma 4C: Sentinel-2 L2A acquisition and bounded reflectance indices.

One user-initiated optical candidate at a time (0–90 days, then 91–180 days). Do not infer a land-cover classification,
thermal temperature or cooling prediction from these optical observations.
"""
from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from datetime import datetime, timedelta, timezone
import math
import re
from typing import Any

import numpy as np
import requests

from satellite_discovery import STAC_SEARCH_URL, _bbox, _parse_datetime
from buffer_geometry import create_analysis_buffer

# Optical observations may characterize surfaces over a longer period than thermal observations.
# Windows are non-overlapping so fallback uses older scenes only when primary fails.
OPTICAL_AGE_TIERS = ((0, 90), (91, 180))
COLLECTION = 'sentinel-2-l2a'
ITEM_URL = 'https://planetarycomputer.microsoft.com/api/stac/v1/collections/sentinel-2-l2a/items/'
REQUIRED_ASSETS = ('B04', 'B08', 'B11', 'SCL')
MAX_SCENE_CLOUD = 80.0
MIN_SITE_VALID_PCT = 70.0  # Preliminary data-quality policy, not model validation
MAX_PIXELS = 1_000_000
# SCL 2 dark, 4 vegetation, 5 non-vegetated, 6 water, 7 unclassified.
# 7 can contain uncertainty, documented in the output.
CLEAR_CLASSES = (2, 4, 5, 6, 7)
SCENE_RE = re.compile(r'^S2[ABC]_MSIL2A_[A-Za-z0-9_]+$')


@dataclass(frozen=True)
class SentinelCandidate:
    item_id: str
    platform: str
    observed_utc: str
    age_days: int
    scene_cloud_pct: float
    tier: str
    processing_baseline: str


@dataclass(frozen=True)
class SentinelDiscovery:
    candidate: SentinelCandidate | None
    searched_tiers: tuple[str, ...]
    searched_at_utc: str


@dataclass(frozen=True)
class IndexSummary:
    scene_id: str
    observed_utc: str
    footprint_pixels: int
    valid_pixels: int
    valid_coverage_pct: float | None
    ndvi_mean: float | None
    ndbi_mean: float | None
    ndvi_p10: float | None
    ndvi_p90: float | None
    status: str
    notes: tuple[str, ...]
    ndvi_overlay_png: bytes | None = None
    ndbi_overlay_png: bytes | None = None
    overlay_bounds: tuple[tuple[float, float], tuple[float, float]] | None = None


def _window(now: datetime, young: int, old: int) -> str:
    start = (now.date() - timedelta(days=old)).isoformat() + 'T00:00:00Z'
    end = (now.date() - timedelta(days=young)).isoformat() + 'T23:59:59Z'
    return f'{start}/{end}'


def candidate_from_item(item: Any, now: datetime, tier: tuple[int, int]) -> SentinelCandidate | None:
    if not isinstance(item, dict) or item.get('collection') != COLLECTION:
        return None
    props, assets = item.get('properties'), item.get('assets')
    if not isinstance(props, dict) or not isinstance(assets, dict):
        return None
    if not all(isinstance(assets.get(name), dict) for name in REQUIRED_ASSETS):
        return None
    platform = str(props.get('platform', '')).lower()
    if platform not in ('sentinel-2a', 'sentinel-2b', 'sentinel-2c'):
        return None
    scene_id = str(item.get('id', ''))
    if not SCENE_RE.fullmatch(scene_id):
        return None
    try:
        baseline = str(props.get('s2:processing_baseline', props.get('processing:version', '')))
        major = int(baseline.split('.')[0])
        cloud = float(props['eo:cloud_cover'])
        observed = _parse_datetime(props['datetime'])
    except (KeyError, ValueError, TypeError, OverflowError):
        return None
    if major < 4 or not math.isfinite(cloud) or not 0 <= cloud <= MAX_SCENE_CLOUD:
        return None
    age = (now.date() - observed.date()).days
    if observed > now or not tier[0] <= age <= tier[1]:
        return None
    return SentinelCandidate(scene_id, platform, observed.isoformat(timespec='seconds'), age,
                             cloud, f'{tier[0]}–{tier[1]} days', baseline)


def discover_sentinel(boundary_geom: dict, buffer_m: int, *, now=None, session=None) -> SentinelDiscovery:
    now = now or datetime.now(timezone.utc)
    if now.tzinfo is None:
        raise ValueError('The reference time must be timezone-aware.')
    now = now.astimezone(timezone.utc)
    bbox = _bbox(boundary_geom, buffer_m)
    client = session or requests
    tiers = []
    for tier in OPTICAL_AGE_TIERS:
        label = f'{tier[0]}–{tier[1]} days'
        tiers.append(label)
        payload = {'collections': [COLLECTION], 'bbox': bbox,
                   'datetime': _window(now, *tier), 'limit': 100}
        try:
            response = client.post(STAC_SEARCH_URL, json=payload, timeout=(7, 25),
                                   headers={'Accept': 'application/geo+json',
                                            'User-Agent': 'ClimaForma/0.4 (user-initiated research demo)'})
            response.raise_for_status()
            data = response.json()
        except (requests.RequestException, ValueError) as exc:
            raise ValueError('Sentinel-2 catalogue unavailable; try again later.') from exc
        if not isinstance(data, dict) or not isinstance(data.get('features'), list):
            raise ValueError('Sentinel-2 catalogue returned invalid data.')
        # This returns the newest candidate in the first eligible tier. A later
        # iteration may need to evaluate multiple tiles if one covers only part of the site.
        qualified = [v for item in data['features'] if (v := candidate_from_item(item, now, tier))]
        if qualified:
            qualified.sort(key=lambda x: (-_parse_datetime(x.observed_utc).timestamp(), x.scene_cloud_pct))
            return SentinelDiscovery(qualified[0], tuple(tiers), now.isoformat(timespec='seconds'))
        # Search pagination must not silently omit newer or better scenes.
        if any(isinstance(x, dict) and x.get('rel') == 'next' for x in data.get('links', [])):
            raise ValueError('Sentinel catalogue returned more candidate pages than this MVP can check; refine the site or retry.')
    return SentinelDiscovery(None, tuple(tiers), now.isoformat(timespec='seconds'))


def compute_indices(red_dn, nir_dn, swir_dn, scl, site_mask, *, offset=-1000.0):
    """Calculate NDVI/NDBI on a co-registered 20 m grid using corrected BOA DNs.

    For current processing baselines >=04.00, reflectance=(DN-1000)/10000;
    the shared factor /10000 cancels in normalized differences, offset does not.
    """
    arrays = [np.asarray(x) for x in (red_dn, nir_dn, swir_dn, scl, site_mask)]
    if any(x.ndim != 2 or x.shape != arrays[0].shape for x in arrays):
        raise ValueError('All aligned Sentinel-2 arrays must have the same 2D shape.')
    red, nir, swir, classes, footprint = arrays
    valid = footprint.astype(bool) & np.isin(classes, CLEAR_CLASSES)
    for band in (red, nir, swir):
        valid &= np.isfinite(band) & (band > 0)
    r = red.astype(np.float64) + offset
    n = nir.astype(np.float64) + offset
    s = swir.astype(np.float64) + offset
    den_ndvi, den_ndbi = n + r, s + n
    valid &= np.isfinite(den_ndvi) & np.isfinite(den_ndbi) & (abs(den_ndvi) > 1) & (abs(den_ndbi) > 1)
    ndvi = np.full(red.shape, np.nan, dtype=np.float32)
    ndbi = np.full(red.shape, np.nan, dtype=np.float32)
    ndvi[valid] = ((n[valid] - r[valid]) / den_ndvi[valid]).astype(np.float32)
    ndbi[valid] = ((s[valid] - n[valid]) / den_ndbi[valid]).astype(np.float32)
    valid &= (abs(ndvi) <= 1) & (abs(ndbi) <= 1)
    ndvi[~valid], ndbi[~valid] = np.nan, np.nan
    return ndvi, ndbi, valid


def fetch_assets(scene: SentinelCandidate, *, session=None, signer=None):
    if not SCENE_RE.fullmatch(scene.item_id):
        raise ValueError('Invalid Sentinel-2 scene ID.')
    client = session or requests
    try:
        response = client.get(ITEM_URL + scene.item_id, timeout=(7, 20))
        response.raise_for_status()
        item = response.json()
    except (requests.RequestException, ValueError) as exc:
        raise ValueError('Could not retrieve the Sentinel-2 scene metadata.') from exc
    if not isinstance(item, dict) or item.get('id') != scene.item_id or item.get('collection') != COLLECTION:
        raise ValueError('The Sentinel-2 item does not match the discovered scene.')
    props = item.get('properties') or {}
    if str(props.get('s2:processing_baseline', props.get('processing:version', ''))) != scene.processing_baseline:
        raise ValueError('Sentinel processing baseline changed; rediscover the scene.')
    assets = item.get('assets') or {}
    urls = {}
    for key in REQUIRED_ASSETS:
        asset = assets.get(key)
        href = asset.get('href') if isinstance(asset, dict) else None
        if not isinstance(href, str) or not href.startswith('https://'):
            raise ValueError(f'Missing secure Sentinel-2 raster asset: {key}.')
        urls[key] = href
    if signer is None:
        import planetary_computer
        signer = planetary_computer.sign
    try:
        return {k: signer(v) for k, v in urls.items()}
    except Exception as exc:
        raise ValueError('Could not sign Sentinel-2 raster URLs.') from exc


def render_index_overlay(values, valid, src_transform, src_crs, *, kind):
    """Warp a masked 20 m index to WGS84 before rendering transparent PNG.

    This is only a map-display reprojection; it creates no new information.
    """
    import rasterio
    from rasterio.enums import Resampling
    from rasterio.warp import calculate_default_transform, reproject, transform_bounds
    from PIL import Image
    values = np.asarray(values, dtype=np.float32)
    valid = np.asarray(valid, dtype=bool) & np.isfinite(values)
    if not valid.any():
        return None, None
    height, width = values.shape
    from rasterio.transform import array_bounds
    west, south, east, north = array_bounds(height, width, src_transform)
    dst_transform, dst_width, dst_height = calculate_default_transform(
        src_crs, 'EPSG:4326', width, height, west, south, east, north)
    if dst_width * dst_height > MAX_PIXELS * 4:
        raise ValueError('Index-map display exceeds safety limits.')
    source_values = np.where(valid, values, -9999.0)
    target = np.full((dst_height, dst_width), -9999.0, dtype=np.float32)
    reproject(source_values, target, src_transform=src_transform, src_crs=src_crs,
              src_nodata=-9999.0, dst_transform=dst_transform, dst_crs='EPSG:4326',
              dst_nodata=-9999.0, resampling=Resampling.nearest)
    target_valid = (target != -9999.0) & np.isfinite(target)
    if kind == 'NDVI':
        # brown -> cream -> green. Fixed scale across observations.
        anchors = np.array([[-1, [130, 84, 59]], [0, [235, 226, 189]], [0.4, [107, 169, 103]], [1, [26, 105, 69]]], dtype=object)
    elif kind == 'NDBI':
        # teal -> cream -> reddish brown. Fixed scale across observations.
        anchors = np.array([[-1, [50, 120, 132]], [0, [233, 232, 208]], [0.4, [180, 103, 70]], [1, [125, 55, 45]]], dtype=object)
    else:
        raise ValueError('Unsupported optical index.')
    x = np.clip(target, -1, 1)
    levels = [float(r[0]) for r in anchors]
    colors = np.stack([np.interp(x, levels, [r[1][j] for r in anchors]) for j in range(3)], axis=-1).astype(np.uint8)
    alpha = np.where(target_valid, 195, 0).astype(np.uint8)
    with BytesIO() as buf:
        Image.fromarray(np.dstack([colors, alpha]), 'RGBA').save(buf, format='PNG')
        png = buf.getvalue()
    from rasterio.transform import array_bounds
    w, s, e, n = array_bounds(dst_height, dst_width, dst_transform)
    return png, ((s, w), (n, e))


def assess_sentinel(scene: SentinelCandidate, boundary_geom: dict, buffer_m: int,
                    *, session=None, signer=None) -> IndexSummary:
    from contextlib import ExitStack
    import rasterio
    from rasterio.enums import Resampling
    from rasterio.features import geometry_mask, geometry_window
    from rasterio.vrt import WarpedVRT
    from rasterio.warp import transform_geom
    urls = fetch_assets(scene, session=session, signer=signer)
    from shapely.geometry import shape, mapping
    site = shape(boundary_geom)
    buffer = create_analysis_buffer(boundary_geom, buffer_m)
    context = shape(buffer.geometry) if buffer else site
    try:
        with ExitStack() as stack:
            datasets = {k: stack.enter_context(rasterio.open(v)) for k, v in urls.items()}
            # Native B11 grid is 20 m; align red/NIR with bilinear, SCL with nearest.
            grid = datasets['B11']
            if grid.crs is None or grid.crs.is_geographic:
                raise ValueError('Expected Sentinel-2 projected 20 m analysis grid.')
            native_site = transform_geom('EPSG:4326', grid.crs, mapping(site))
            native_context = transform_geom('EPSG:4326', grid.crs, mapping(context))
            win = geometry_window(grid, [native_context], pixel_precision=3)
            win = win.round_offsets().round_lengths()
            if win.width <= 0 or win.height <= 0 or win.width * win.height > MAX_PIXELS:
                raise ValueError('Sentinel-2 site buffer is outside scene or exceeds raster safety limit.')
            transform = grid.window_transform(win)
            footprint = geometry_mask([native_site], out_shape=(int(win.height), int(win.width)),
                                      transform=transform, invert=True, all_touched=False)
            aligned = {}
            for name, dataset in datasets.items():
                if name == 'B11':
                    aligned[name] = dataset.read(1, window=win)
                else:
                    vrt = stack.enter_context(WarpedVRT(
                        dataset, crs=grid.crs, transform=grid.transform,
                        width=grid.width, height=grid.height,
                        resampling=Resampling.nearest if name == 'SCL' else Resampling.bilinear))
                    aligned[name] = vrt.read(1, window=win)
            ndvi, ndbi, valid = compute_indices(
                aligned['B04'], aligned['B08'], aligned['B11'], aligned['SCL'], footprint)
            total, good = int(footprint.sum()), int(valid.sum())
            ndvi_png, ndvi_bounds = render_index_overlay(ndvi, valid, transform, grid.crs, kind='NDVI')
            ndbi_png, ndbi_bounds = render_index_overlay(ndbi, valid, transform, grid.crs, kind='NDBI')
    except ValueError:
        raise
    except Exception as exc:
        raise ValueError('Unable to read Sentinel-2 COG windows. Check remote raster access.') from exc
    coverage = round(100 * good / total, 1) if total else None
    status = ('INSUFFICIENT_SITE_PIXELS' if not total else
              'LOW_VALID_SITE_COVERAGE' if coverage < MIN_SITE_VALID_PCT else
              'PRELIMINARY_QA_PASS')
    samples = ndvi[valid]
    built = ndbi[valid]
    return IndexSummary(scene.item_id, scene.observed_utc, total, good, coverage,
                        round(float(np.mean(samples)), 3) if good else None,
                        round(float(np.mean(built)), 3) if good else None,
                        round(float(np.percentile(samples, 10)), 3) if good else None,
                        round(float(np.percentile(samples, 90)), 3) if good else None,
                        status, (
                            'Sentinel-2 MSI L2A optical reflectance; this is not a thermal product.',
                            'NDVI/NDBI on a 20 m co-registered analysis grid; not a land-cover classification.',
                            'Processing baseline >=04.00: BOA_ADD_OFFSET=-1000, quantification=10000.',
                            'SCL classes 2,4,5,6,7 retained; class 7 is unclassified and uncertain.',
                            'One satellite tile and one scene only; no compositing, fallback after site QA failure, or independent validation.',
                        ), ndvi_png, ndbi_png, ndvi_bounds)
