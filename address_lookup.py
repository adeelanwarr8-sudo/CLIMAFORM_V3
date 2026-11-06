"""Explicit, bounded address search for ClimaForma's small research prototype.

Public OSMF Nominatim: no autocomplete, no bulk search, requests cached and
rate-limited to <= 1 request/second per running Python process.
For production/multi-instance deployment use a contracted geocoding service.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
import math
import os
import threading
import time

import requests

API_URL = "https://nominatim.openstreetmap.org/search"
HEADERS = {"User-Agent": "ClimaFormaResearchPrototype/0.4 (single-user address navigation)",
           "Accept": "application/json"}
_MAX_CACHE = 256
_CACHE_TTL = timedelta(hours=24)
_lock = threading.Lock()
_cache: dict[str, tuple[datetime, list[dict]]] = {}
_last_request = 0.0


def _parse_results(data: object) -> list[dict]:
    if not isinstance(data, list):
        raise ValueError("Address service returned an invalid response.")
    results = []
    for item in data[:5]:
        if not isinstance(item, dict):
            continue
        try:
            lat, lon = float(item["lat"]), float(item["lon"])
            label = str(item["display_name"]).strip()
        except (ValueError, KeyError, TypeError):
            continue
        if label and math.isfinite(lat) and math.isfinite(lon) and -90 <= lat <= 90 and -180 <= lon <= 180:
            results.append({"lat": lat, "lon": lon, "label": label[:300]})
    return results


def search_addresses(query: str) -> list[dict]:
    """Search only after user presses Search. Cache by normalized query."""
    global _last_request
    if not isinstance(query, str) or not 3 <= len(query.strip()) <= 180:
        raise ValueError("Enter at least three characters (maximum 180) to search for a place.")
    key = " ".join(query.strip().casefold().split())
    with _lock:
        now = datetime.now(timezone.utc)
        cached = _cache.get(key)
        if cached and now - cached[0] < _CACHE_TTL:
            return [dict(r) for r in cached[1]]
        if cached:
            del _cache[key]
        elapsed = time.monotonic() - _last_request
        if elapsed < 1.1:
            time.sleep(1.1 - elapsed)
        _last_request = time.monotonic()
        headers = dict(HEADERS)
        contact = os.getenv("CLIMAFORMA_CONTACT_EMAIL", "").strip()
        if contact:
            headers["User-Agent"] += f" (contact: {contact})"
        try:
            response = requests.get(API_URL, params={"q": query.strip(), "format": "jsonv2", "limit": 5},
                                    headers=headers, timeout=(5, 12))
            response.raise_for_status()
            results = _parse_results(response.json())
        except (requests.RequestException, ValueError) as exc:
            raise ValueError("Address lookup unavailable or rate-limited; retry later or navigate manually.") from exc
        if len(_cache) >= _MAX_CACHE:
            _cache.pop(next(iter(_cache)))
        _cache[key] = (datetime.now(timezone.utc), results)
        return [dict(r) for r in results]
