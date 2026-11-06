"""ClimaForma 4B: bounded Landsat L2 surface-temperature and QA_PIXEL inspection.

Reads the selected STAC scene's signed Cloud Optimized GeoTIFFs on demand.
Reports coverage and QA only; does not calculate or display temperatures.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable
import math
import re

import numpy as np
import requests
from shapely.geometry import shape, mapping

from buffer_geometry import create_analysis_buffer

ITEM_URL = 'https://planetarycomputer.microsoft.com/api/stac/v1/collections/landsat-c2-l2/items/'
SCENE_ID_RE = re.compile(r'^LC0[89]_L2SP_[A-Za-z0-9_]+$')
MAX_WINDOW_PIXELS = 1_000_000
MIN_SITE_VALID_PERCENT = 70.0  # Provisional acquisition-quality criterion, not scientific validation.
# Landsat C2 QA_PIXEL: fill 0, dilated cloud 1, cirrus 2, cloud 3,
# cloud shadow 4, snow 5. Water bit 7 is not inherently invalid for LST.
BAD_QA_BITS = sum(1 << b for b in (0, 1, 2, 3, 4, 5))


@dataclass(frozen=True)
class Coverage:
    footprint_pixels: int
    valid_pixels: int
    coverage_pct: float | None


@dataclass(frozen=True)
class RasterAssessment:
    scene_id: str
    site: Coverage
    context: Coverage
    pixel_size_m: tuple[float, float]
    status: str
    notes: tuple[str, ...]


def valid_pixel_mask(st_dn: np.ndarray, qa_pixel: np.ndarray,
                     st_nodata: float | None = 0) -> np.ndarray:
    """Raw DN/QA validation (no LST conversion and no resampling-based claims)."""
    if st_dn.shape != qa_pixel.shape or st_dn.ndim != 2:
        raise ValueError('Thermal and QA rasters must share the same 2D grid.')
    valid_st = np.isfinite(st_dn) & (st_dn > 0)
    if st_nodata is not None:
        valid_st &= st_dn != st_nodata
    qa_int = qa_pixel.astype(np.uint16, copy=False)
    return valid_st & (qa_int & BAD_QA_BITS == 0)


def _coverage(footprint: np.ndarray, valid: np.ndarray) -> Coverage:
    total = int(footprint.sum())
    accepted = int((footprint & valid).sum())
    return Coverage(total, accepted, round(100 * accepted / total, 1) if total else None)


def evaluate_rasters(st_path: str, qa_path: str, boundary_geom: dict,
                     buffer_m: int = 100, *, scene_id: str = 'local-test') -> RasterAssessment:
    """Read a small window on the thermal raster's native delivery grid.

    QA is nearest-neighbour aligned to that grid where required. Pixel centres
    inside the polygon are counted. Overlap fractions and independent 100 m
    thermal resolution are NOT inferred from the 30 m delivery grid.
    """
    import rasterio
    from rasterio.enums import Resampling
    from rasterio.features import geometry_mask
    from rasterio.vrt import WarpedVRT
    from rasterio.warp import transform_geom
    from rasterio.windows import Window, from_bounds, transform as window_transform

    try:
        site = shape(boundary_geom)
        if site.geom_type != 'Polygon' or site.is_empty or not site.is_valid:
            raise ValueError('A valid site polygon is required.')
        buffer = create_analysis_buffer(boundary_geom, buffer_m)
        context = shape(buffer.geometry) if buffer is not None else site
    except (TypeError, KeyError) as exc:
        raise ValueError('Invalid site geometry or buffer.') from exc

    try:
        with rasterio.Env(GDAL_DISABLE_READDIR_ON_OPEN='EMPTY_DIR'):
            with rasterio.open(st_path) as st_ds, rasterio.open(qa_path) as qa_ds:
                if not st_ds.crs or not qa_ds.crs:
                    raise ValueError('Thermal and QA assets must have a defined CRS.')
                site_native = transform_geom('EPSG:4326', st_ds.crs, mapping(site))
                context_native = transform_geom('EPSG:4326', st_ds.crs, mapping(context))
                context_bounds = shape(context_native).bounds
                raw_window = from_bounds(*context_bounds, transform=st_ds.transform)
                full = Window(0, 0, st_ds.width, st_ds.height)
                try:
                    window = raw_window.round_offsets(op='floor').round_lengths(op='ceil').intersection(full)
                except Exception as exc:
                    raise ValueError('Scene thermal raster does not overlap the selected site.') from exc
                window = Window(int(window.col_off), int(window.row_off), int(window.width), int(window.height))
                if window.width < 1 or window.height < 1:
                    raise ValueError('No thermal pixels intersect the site context.')
                if window.width * window.height > MAX_WINDOW_PIXELS:
                    raise ValueError('Thermal raster request exceeds the one-million-pixel safety limit.')
                st_dn = st_ds.read(1, window=window, masked=False)
                with WarpedVRT(qa_ds, crs=st_ds.crs, transform=st_ds.transform,
                               width=st_ds.width, height=st_ds.height,
                               resampling=Resampling.nearest) as qa_aligned:
                    qa = qa_aligned.read(1, window=window, masked=False)
                transform = window_transform(window, st_ds.transform)
                footprint_site = geometry_mask([site_native], out_shape=st_dn.shape,
                                               transform=transform, invert=True, all_touched=False)
                footprint_context = geometry_mask([context_native], out_shape=st_dn.shape,
                                                  transform=transform, invert=True, all_touched=False)
                valid = valid_pixel_mask(st_dn, qa, st_ds.nodata)
                site_result = _coverage(footprint_site, valid)
                context_result = _coverage(footprint_context, valid)
                resolution = (abs(st_ds.res[0]), abs(st_ds.res[1]))
                if st_ds.crs.is_geographic:
                    resolution = (float('nan'), float('nan'))
    except ValueError:
        raise
    except Exception as exc:
        raise ValueError('Could not read signed Landsat thermal/QA assets. Check remote COG access and retry.') from exc

    if site_result.footprint_pixels == 0:
        status = 'INSUFFICIENT_SITE_PIXELS'
    elif site_result.coverage_pct is not None and site_result.coverage_pct < MIN_SITE_VALID_PERCENT:
        status = 'LOW_VALID_SITE_COVERAGE'
    else:
        status = 'PRELIMINARY_QA_PASS'
    return RasterAssessment(scene_id=scene_id, site=site_result, context=context_result,
                            pixel_size_m=resolution, status=status, notes=(
                                'Preliminary QA only; no surface temperature is calculated.',
                                'Coverage counts centres of 30 m delivery pixels, not independent 100 m thermal observations.',
                                f'Provisional site valid-coverage screening threshold: {MIN_SITE_VALID_PERCENT:.0f}%.',
                                'Site and site-plus-buffer coverage are reported separately.',
                            ))


def fetch_item_assets(scene_id: str, *, session=None, signer: Callable[[str], str] | None = None) -> tuple[str, str]:
    """Obtain signed scene assets; never trust an arbitrary user-provided URL."""
    if not isinstance(scene_id, str) or not SCENE_ID_RE.fullmatch(scene_id):
        raise ValueError('Invalid Landsat 8/9 surface-temperature scene ID.')
    client = session or requests
    try:
        response = client.get(ITEM_URL + scene_id, timeout=(7, 20),
                              headers={'Accept': 'application/geo+json'})
        response.raise_for_status()
        item = response.json()
    except (requests.RequestException, ValueError) as exc:
        raise ValueError('Could not retrieve the Landsat STAC item. Please retry.') from exc
    if not isinstance(item, dict) or item.get('id') != scene_id or item.get('collection') != 'landsat-c2-l2':
        raise ValueError('Satellite catalogue returned a mismatched STAC item.')
    assets = item.get('assets') or {}
    if not isinstance(assets, dict):
        raise ValueError('STAC item has invalid assets.')
    urls = []
    for key in ('lwir11', 'qa_pixel'):
        entry = assets.get(key)
        href = entry.get('href') if isinstance(entry, dict) else None
        if not isinstance(href, str) or not href.startswith('https://'):
            raise ValueError(f'Missing secure Landsat asset: {key}.')
        urls.append(href)
    if signer is None:
        try:
            import planetary_computer
        except ImportError as exc:
            raise ValueError('Install planetary-computer for secure asset URL signing.') from exc
        signer = planetary_computer.sign
    try:
        return tuple(signer(url) for url in urls)
    except Exception as exc:
        raise ValueError('Unable to obtain access to the Landsat raster assets.') from exc


def assess_candidate(candidate, boundary_geom: dict, buffer_m: int,
                     *, session=None, signer=None) -> RasterAssessment:
    st_url, qa_url = fetch_item_assets(candidate.item_id, session=session, signer=signer)
    return evaluate_rasters(st_url, qa_url, boundary_geom, buffer_m, scene_id=candidate.item_id)
