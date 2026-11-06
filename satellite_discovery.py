"""ClimaForma Increment 4A: metadata-only Landsat 8/9 scene discovery.

Searches public Microsoft Planetary Computer STAC API on demand. Does NOT
retrieve raster pixels, test site-specific clouds, or calculate temperatures.
"""
from __future__ import annotations

from datetime import datetime, date, time, timedelta, timezone
from dataclasses import dataclass
from typing import Any
import math
import requests

STAC_SEARCH_URL = 'https://planetarycomputer.microsoft.com/api/stac/v1/search'
COLLECTION = 'landsat-c2-l2'
# Search only these non-overlapping acquisition-age ranges, in priority order.
AGE_TIERS = ((3, 7), (8, 15), (16, 30), (31, 45))
MAX_SCENE_CLOUD = 80.0  # preliminary scene-wide catalogue filter, NOT site quality
MAX_PAGES_PER_TIER = 4
PAGE_SIZE = 100


@dataclass(frozen=True)
class LandsatCandidate:
    item_id: str
    platform: str
    observed_utc: str
    age_days: int
    scene_cloud_pct: float
    tier: str
    thermal_asset: str
    collection: str = COLLECTION


@dataclass(frozen=True)
class DiscoveryResult:
    candidate: LandsatCandidate | None
    searched_at_utc: str
    searched_tiers: tuple[str, ...]
    notes: tuple[str, ...]


def _parse_datetime(value: str) -> datetime:
    if not isinstance(value, str):
        raise ValueError('Missing acquisition timestamp')
    parsed = datetime.fromisoformat(value.replace('Z', '+00:00'))
    if parsed.tzinfo is None:
        raise ValueError('Acquisition timestamp must be timezone-aware')
    return parsed.astimezone(timezone.utc)


def _age(observed: datetime, now: datetime) -> int:
    return (now.date() - observed.date()).days


def _tier_window(now: datetime, youngest: int, oldest: int) -> str:
    # Includes all UTC acquisition times on each date, with non-overlapping tiers.
    start = (now.date() - timedelta(days=oldest)).isoformat() + 'T00:00:00Z'
    end = (now.date() - timedelta(days=youngest)).isoformat() + 'T23:59:59Z'
    return f'{start}/{end}'


def _metadata_candidate(item: Any, now: datetime, tier: tuple[int, int]) -> LandsatCandidate | None:
    if not isinstance(item, dict) or item.get('collection') != COLLECTION:
        return None
    props = item.get('properties') or {}
    assets = item.get('assets') or {}
    if not isinstance(props, dict) or not isinstance(assets, dict):
        return None
    platform = str(props.get('platform', '')).lower()
    if platform not in ('landsat-8', 'landsat-9'):
        return None
    item_id = str(item.get('id', ''))
    # L2SP indicates that the item includes a surface-temperature product.
    if '_L2SP_' not in item_id:
        return None
    if not all(isinstance(assets.get(key), dict) for key in ('lwir11', 'qa_pixel')):
        return None
    cloud = props.get('eo:cloud_cover')
    try:
        cloud = float(cloud)
    except (ValueError, TypeError):
        return None
    if not math.isfinite(cloud) or not (0 <= cloud <= MAX_SCENE_CLOUD):
        return None
    try:
        observed = _parse_datetime(props.get('datetime'))
    except (ValueError, TypeError, OverflowError):
        return None
    age = _age(observed, now)
    if not tier[0] <= age <= tier[1] or observed > now:
        return None
    return LandsatCandidate(item_id, platform, observed.isoformat(timespec='seconds'),
                            age, cloud, f'{tier[0]}–{tier[1]} days', 'lwir11')


def _bbox(boundary_geom: dict, buffer_m: int) -> list[float]:
    from shapely.geometry import shape
    from buffer_geometry import create_analysis_buffer
    if not isinstance(boundary_geom, dict) or boundary_geom.get('type') != 'Polygon':
        raise ValueError('Draw or select a valid site boundary first.')
    site = shape(boundary_geom)
    if site.is_empty or not site.is_valid:
        raise ValueError('The selected site boundary is invalid.')
    context = create_analysis_buffer(boundary_geom, buffer_m)
    geom = shape(context.geometry) if context is not None else site
    values = list(geom.bounds)
    if any(not math.isfinite(x) for x in values):
        raise ValueError('Invalid geographic search extent.')
    if not (-180 <= values[0] < values[2] <= 180 and -90 <= values[1] < values[3] <= 90):
        raise ValueError('Satellite search extent must be valid WGS84 coordinates.')
    return values


def discover_landsat(boundary_geom: dict, buffer_m: int, *, now: datetime | None = None,
                     session: Any = None) -> DiscoveryResult:
    """Search successive tiers. Stop at first tier with an eligible metadata candidate.

    Public STAC availability, catalogue publication delay and scene-level cloud
    metadata do not establish site-specific usable thermal-pixel coverage.
    """
    now = now or datetime.now(timezone.utc)
    if now.tzinfo is None:
        raise ValueError('Current time must include a timezone.')
    now = now.astimezone(timezone.utc)
    bbox = _bbox(boundary_geom, buffer_m)
    client = session if session is not None else requests
    searched = []
    for tier in AGE_TIERS:
        label = f'{tier[0]}–{tier[1]} days'
        searched.append(label)
        payload = {'collections': [COLLECTION], 'bbox': bbox,
                   'datetime': _tier_window(now, *tier), 'limit': PAGE_SIZE}
        best = []
        next_url = STAC_SEARCH_URL
        for page in range(MAX_PAGES_PER_TIER):
            try:
                response = client.post(next_url, json=payload, timeout=(7, 22),
                                       headers={'Accept': 'application/geo+json',
                                                'User-Agent': 'ClimaForma/0.4 (user-initiated research prototype)'})
                response.raise_for_status()
                result = response.json()
            except (requests.RequestException, ValueError) as exc:
                raise ValueError('Satellite catalogue request failed. Retry later; no observation has been selected.') from exc
            if not isinstance(result, dict) or not isinstance(result.get('features'), list):
                raise ValueError('Satellite catalogue returned an invalid STAC response.')
            for item in result['features']:
                candidate = _metadata_candidate(item, now, tier)
                if candidate is not None:
                    best.append(candidate)
            links = result.get('links') or []
            next_link = next((x for x in links if isinstance(x, dict) and x.get('rel') == 'next'), None)
            # Follow POST pagination only, and only on the same trusted STAC host.
            if next_link and page + 1 < MAX_PAGES_PER_TIER:
                from urllib.parse import urlparse
                link_url = next_link.get('href', '')
                if urlparse(link_url).netloc != urlparse(STAC_SEARCH_URL).netloc:
                    raise ValueError('Catalogue pagination returned an unexpected host.')
                next_url = link_url
                method = str(next_link.get('method', 'GET')).upper()
                if method != 'POST':
                    # Avoid opening arbitrary GET URLs through a POST-only client.
                    break
                payload = next_link.get('body', payload)
                if not isinstance(payload, dict):
                    raise ValueError('Invalid satellite catalogue pagination request.')
            else:
                break
        if best:
            # Prefer newest scene; break same-day ties on lower cloud cover.
            best.sort(key=lambda c: (-_parse_datetime(c.observed_utc).timestamp(), c.scene_cloud_pct))
            return DiscoveryResult(best[0], now.isoformat(timespec='seconds'), tuple(searched),
                                   ('Metadata-qualified only; scene-wide cloud filter ≤80%.',
                                    'Site cloud, pixel validity and thermal raster not yet checked.',
                                    'Observation age uses UTC acquisition date; never catalogue publication date.'))
    return DiscoveryResult(None, now.isoformat(timespec='seconds'), tuple(searched),
                           ('No metadata-qualified Landsat 8/9 surface-temperature scene in the permitted 3–45-day windows.',
                            'Scenes aged 0–2 days and older than 45 days were not searched.',
                            'No historical fallback or substitute values used.'))
