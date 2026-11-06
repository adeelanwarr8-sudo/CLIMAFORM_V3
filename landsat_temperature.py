"""Increment 4D: observed Landsat C2 L2 surface temperature, not downscaled prediction.

Reads signed ST_B10 (Planetary Computer `lwir11`) and QA_PIXEL directly from
one selected scene. Statistics use valid 30 m *delivery-grid* pixel centres.
Native thermal information is ~100 m: this is not courtyard-scale precision.
"""
from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
import math
import numpy as np
from PIL import Image
from shapely.geometry import shape, mapping
from buffer_geometry import create_analysis_buffer
from landsat_raster import fetch_item_assets, valid_pixel_mask, MAX_WINDOW_PIXELS, MIN_SITE_VALID_PERCENT

ST_SCALE = 0.00341802
ST_OFFSET_K = 149.0
KELVIN_TO_CELSIUS = 273.15


@dataclass(frozen=True)
class TemperatureAssessment:
    scene_id: str
    observed_utc: str
    site_pixels: int
    valid_site_pixels: int
    site_coverage_pct: float
    context_pixels: int
    valid_context_pixels: int
    context_coverage_pct: float
    mean_c: float | None
    p95_c: float | None
    minimum_c: float | None
    maximum_c: float | None
    status: str
    overlay_png: bytes | None
    overlay_bounds: tuple[tuple[float, float], tuple[float, float]] | None
    display_range_c: tuple[float, float] | None
    notes: tuple[str, ...]


def kelvin_dn_to_celsius(dn: np.ndarray) -> np.ndarray:
    """USGS Collection 2 Level-2 ST scale/offset; DN=0 is fill and must be masked."""
    return np.asarray(dn, dtype=np.float64) * ST_SCALE + ST_OFFSET_K - KELVIN_TO_CELSIUS


def _rgba_png(values: np.ndarray, valid: np.ndarray, lo: float, hi: float) -> bytes:
    """Color an observed raster with transparency for invalid/outside-site cells."""
    if not math.isfinite(lo) or not math.isfinite(hi) or hi <= lo:
        raise ValueError('Invalid temperature display range.')
    x = np.clip((np.nan_to_num(values, nan=lo) - lo) / (hi - lo), 0, 1)
    # Blue–cyan–yellow–red: cartographic rendering only; not data transformation.
    colors = np.array([[44, 94, 162], [62, 170, 192], [245, 217, 105], [211, 72, 55]], dtype=np.float64)
    pos = np.linspace(0, 1, len(colors))
    rgb = np.stack([np.interp(x, pos, colors[:, i]) for i in range(3)], axis=-1).astype(np.uint8)
    alpha = np.where(valid, 185, 0).astype(np.uint8)
    rgba = np.dstack([rgb, alpha])
    with BytesIO() as buffer:
        Image.fromarray(rgba, 'RGBA').save(buffer, format='PNG')
        return buffer.getvalue()


def process_temperature(st_path: str, qa_path: str, boundary_geom: dict, buffer_m: int,
                        *, scene_id: str, observed_utc: str) -> TemperatureAssessment:
    import rasterio
    from rasterio.enums import Resampling
    from rasterio.features import geometry_mask
    from rasterio.vrt import WarpedVRT
    from rasterio.warp import transform_geom, calculate_default_transform, reproject
    from rasterio.windows import Window, from_bounds, transform as window_transform, bounds as window_bounds

    site = shape(boundary_geom)
    if site.geom_type != 'Polygon' or site.is_empty or not site.is_valid:
        raise ValueError('A valid site polygon is required.')
    buffer = create_analysis_buffer(boundary_geom, buffer_m)
    context = shape(buffer.geometry) if buffer is not None else site
    try:
        with rasterio.Env(GDAL_DISABLE_READDIR_ON_OPEN='EMPTY_DIR'):
            with rasterio.open(st_path) as ds, rasterio.open(qa_path) as qa_ds:
                if not ds.crs or not qa_ds.crs:
                    raise ValueError('Landsat assets require declared coordinate reference systems.')
                site_native = transform_geom('EPSG:4326', ds.crs, mapping(site))
                context_native = transform_geom('EPSG:4326', ds.crs, mapping(context))
                raw = from_bounds(*shape(context_native).bounds, transform=ds.transform)
                try:
                    win = raw.round_offsets(op='floor').round_lengths(op='ceil').intersection(Window(0, 0, ds.width, ds.height))
                except Exception as exc:
                    raise ValueError('Selected scene does not overlap the project site.') from exc
                win = Window(int(win.col_off), int(win.row_off), int(win.width), int(win.height))
                if win.width * win.height > MAX_WINDOW_PIXELS:
                    raise ValueError('Raster window exceeds safe processing size.')
                if not win.width or not win.height:
                    raise ValueError('No thermal raster cells overlap the site.')
                dn = ds.read(1, window=win, masked=False)
                with WarpedVRT(qa_ds, crs=ds.crs, transform=ds.transform, width=ds.width,
                               height=ds.height, resampling=Resampling.nearest) as qa_vrt:
                    qa = qa_vrt.read(1, window=win, masked=False)
                affine = window_transform(win, ds.transform)
                site_mask = geometry_mask([site_native], out_shape=dn.shape, transform=affine,
                                          invert=True, all_touched=False)
                context_mask = geometry_mask([context_native], out_shape=dn.shape, transform=affine,
                                             invert=True, all_touched=False)
                valid = valid_pixel_mask(dn, qa, ds.nodata)
                valid_site = valid & site_mask
                valid_context = valid & context_mask
                n_site, n_context = int(site_mask.sum()), int(context_mask.sum())
                v_site, v_context = int(valid_site.sum()), int(valid_context.sum())
                p_site = 100 * v_site / n_site if n_site else 0.0
                p_context = 100 * v_context / n_context if n_context else 0.0
                good = n_site > 0 and p_site >= MIN_SITE_VALID_PERCENT and v_site > 0
                vals = kelvin_dn_to_celsius(dn)
                site_values = vals[valid_site]
                mean = float(np.mean(site_values)) if good else None
                p95 = float(np.percentile(site_values, 95)) if good else None
                vmin = float(np.min(site_values)) if good else None
                vmax = float(np.max(site_values)) if good else None
                png, bounds, display_range = None, None, None
                if good:
                    # Compute visualization range from real valid context observations.
                    context_values = vals[valid_context]
                    low, high = (float(np.percentile(context_values, 2)),
                                 float(np.percentile(context_values, 98)))
                    if high <= low: low, high = low - 0.5, high + 0.5
                    display_range = (round(low, 1), round(high, 1))
                    # Reproject once to geographic CRS. Never use resampling to infer
                    # additional independent thermal observations or site statistics.
                    left, bottom, right, top = window_bounds(win, ds.transform)
                    out_tf, w, h = calculate_default_transform(ds.crs, 'EPSG:4326',
                                                                int(win.width), int(win.height),
                                                                left, bottom, right, top)
                    if w * h > MAX_WINDOW_PIXELS:
                        raise ValueError('Geographic display exceeds safety pixel limit.')
                    source = np.where(valid_context, vals, -9999.0).astype(np.float32)
                    target = np.full((h, w), -9999.0, dtype=np.float32)
                    reproject(source, target, src_transform=affine, src_crs=ds.crs,
                              dst_transform=out_tf, dst_crs='EPSG:4326',
                              src_nodata=-9999.0, dst_nodata=-9999.0,
                              resampling=Resampling.nearest)
                    show = np.isfinite(target) & (target != -9999)
                    png = _rgba_png(target, show, low, high)
                    west, north = out_tf * (0, 0)
                    east, south = out_tf * (w, h)
                    bounds = ((float(south), float(west)), (float(north), float(east)))
    except ValueError:
        raise
    except Exception as exc:
        raise ValueError('Could not process signed Landsat ST and QA rasters. Retry the selected scene.') from exc
    status = ('SITE_QA_PASS' if good else
              'INSUFFICIENT_SITE_PIXELS' if not n_site else 'LOW_VALID_SITE_COVERAGE')
    return TemperatureAssessment(
        scene_id, observed_utc, n_site, v_site, round(p_site, 1), n_context, v_context,
        round(p_context, 1), mean, p95, vmin, vmax, status, png, bounds, display_range,
        ('Observed Landsat Level-2 surface temperature; not air temperature or thermal comfort.',
         '30 m delivery grid; native thermal information approximately 100 m.',
         'Statistics are pixel-centre estimates and are not independent 30 m thermal measurements.',
         'Single overpass only; results are not daily maxima or intervention cooling predictions.',
         'Site statistics withheld unless provisional site QA threshold is met.'))


def assess_temperature_candidate(candidate, boundary_geom: dict, buffer_m: int,
                                 *, session=None, signer=None) -> TemperatureAssessment:
    st_url, qa_url = fetch_item_assets(candidate.item_id, session=session, signer=signer)
    return process_temperature(st_url, qa_url, boundary_geom, buffer_m,
                               scene_id=candidate.item_id, observed_utc=candidate.observed_utc)
