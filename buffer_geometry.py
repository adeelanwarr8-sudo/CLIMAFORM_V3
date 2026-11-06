"""ClimaForma Increment 3.1: local-metric analysis context buffer.

Input and output are WGS84 longitude/latitude GeoJSON. Buffer distances are
metres in a local azimuthal equidistant projection centered at the selected
site; the project site itself is never altered.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

from pyproj import CRS, Transformer
from shapely.geometry import shape, mapping
from shapely.ops import transform

BUFFER_MIN_M = 0
BUFFER_MAX_M = 300
BUFFER_DEFAULT_M = 25
BUFFER_STEP_M = 25


@dataclass(frozen=True)
class AnalysisBuffer:
    distance_m: int
    geometry: dict
    area_m2: float


def create_analysis_buffer(site_geometry: dict, distance_m: int) -> AnalysisBuffer | None:
    """Buffer one valid polygon in metres, returning a WGS84 context polygon.

    Zero means no surrounding analysis region. The returned polygon includes
    the original site as well as its surrounding area, as GIS buffers do.
    """
    if isinstance(distance_m, bool) or not isinstance(distance_m, int) or distance_m not in range(0, 301, 25):
        raise ValueError('Buffer must be 0–300 m in 25 m increments.')
    if distance_m == 0:
        return None
    if not isinstance(site_geometry, dict) or site_geometry.get('type') != 'Polygon':
        raise ValueError('A valid site polygon is required.')
    try:
        site = shape(site_geometry)
    except (ValueError, TypeError, KeyError, IndexError) as exc:
        raise ValueError('Invalid site geometry.') from exc
    if site.is_empty or not site.is_valid or site.area <= 0:
        raise ValueError('A valid nonempty site polygon is required.')
    lon, lat = float(site.centroid.x), float(site.centroid.y)
    if not all(math.isfinite(x) for x in (lon, lat)) or not (-180 <= lon <= 180 and -90 <= lat <= 90):
        raise ValueError('Site coordinates are outside WGS84 bounds.')
    local = CRS.from_proj4(f'+proj=aeqd +lat_0={lat:.12f} +lon_0={lon:.12f} +datum=WGS84 +units=m +no_defs')
    to_local = Transformer.from_crs('EPSG:4326', local, always_xy=True).transform
    to_geo = Transformer.from_crs(local, 'EPSG:4326', always_xy=True).transform
    metric_site = transform(to_local, site)
    buffered_metric = metric_site.buffer(distance_m, quad_segs=12)
    if buffered_metric.is_empty or not buffered_metric.is_valid:
        raise ValueError('Could not construct a valid analysis buffer.')
    buffered_geo = transform(to_geo, buffered_metric)
    return AnalysisBuffer(distance_m=distance_m, geometry=mapping(buffered_geo), area_m2=float(buffered_metric.area))
