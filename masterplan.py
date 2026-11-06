"""ClimaForma v3.0: safe, explicit manual master-plan feature inventory.

This module deliberately performs NO automatic semantic classification or microclimate prediction.
"""
from __future__ import annotations
from copy import deepcopy
import json
import math
from hashlib import sha256
from collections import Counter, defaultdict
from shapely.geometry import shape, mapping
from shapely.validation import explain_validity
from pyproj import Geod

GEOD = Geod(ellps="WGS84")
MAX_BYTES = 5 * 1024 * 1024
MAX_FEATURES = 2000
CLASSES = (
    "Building / roof", "Tree / canopy", "Grass / planting", "Road",
    "Walkway / paved surface", "Other hardscape", "Water / pond",
    "Bare ground", "Other / specify",
)
PALETTE = {
    None: "#94A3B8", "Building / roof": "#8B98A9", "Tree / canopy": "#24785D",
    "Grass / planting": "#79B68B", "Road": "#B58B70", "Walkway / paved surface": "#D3AC76",
    "Other hardscape": "#9B91A9", "Water / pond": "#4499BB", "Bare ground": "#C1A281",
    "Other / specify": "#7F8A9C",
}


def parse_geojson(raw: bytes) -> tuple[list[dict], str]:
    """Return normalized individual features, each UNCLASSIFIED.

    Only WGS84 lon/lat GeoJSON is accepted. Z ordinates are removed for the
    2D prototype. Feature properties are retained but never trusted as classes.
    """
    if not raw or len(raw) > MAX_BYTES:
        raise ValueError("Upload a nonempty GeoJSON file of at most 5 MB.")
    try:
        data = json.loads(raw.decode("utf-8-sig"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("File must be valid UTF-8 GeoJSON.") from exc
    if not isinstance(data, dict):
        raise ValueError("GeoJSON must be an object.")
    crs = data.get("crs")
    if crs:
        name = str((crs.get("properties") or {}).get("name", "")).upper() if isinstance(crs, dict) else ""
        if name not in ("OGC:CRS84", "EPSG:4326", "URN:OGC:DEF:CRS:OGC:1.3:CRS84", "URN:OGC:DEF:CRS:EPSG::4326"):
            raise ValueError("Only WGS84 longitude/latitude GeoJSON (CRS84/EPSG:4326) is supported in this UI prototype.")
    typ = data.get("type")
    if typ == "FeatureCollection":
        entries = data.get("features")
    elif typ == "Feature":
        entries = [data]
    else:
        raise ValueError("Upload a GeoJSON FeatureCollection or Feature.")
    if not isinstance(entries, list) or not (1 <= len(entries) <= MAX_FEATURES):
        raise ValueError(f"The plan must contain 1–{MAX_FEATURES} features.")
    features = []
    for i, feature in enumerate(entries, 1):
        if not isinstance(feature, dict) or feature.get("type") != "Feature" or not isinstance(feature.get("geometry"), dict):
            raise ValueError(f"Feature {i}: invalid GeoJSON feature.")
        try:
            geom = shape(feature["geometry"])
        except Exception as exc:
            raise ValueError(f"Feature {i}: invalid geometry coordinates.") from exc
        if geom.is_empty or not geom.is_valid:
            raise ValueError(f"Feature {i}: invalid geometry ({explain_validity(geom)}).")
        if geom.geom_type not in ("Polygon", "MultiPolygon", "Point", "MultiPoint", "LineString", "MultiLineString"):
            raise ValueError(f"Feature {i}: {geom.geom_type} is unsupported. Split geometry collections into features.")
        if len(geom.bounds) != 4 or not all(math.isfinite(v) for v in geom.bounds):
            raise ValueError(f"Feature {i}: geometry has invalid coordinate bounds.")
        w, s, e, n = geom.bounds
        if w < -180 or e > 180 or s < -90 or n > 90:
            raise ValueError(f"Feature {i}: geometry must use WGS84 longitude/latitude coordinates.")
        # Shapely 2D normalization so elevations do not break Folium editing/measurements.
        from shapely import force_2d
        geom = force_2d(geom)
        props = deepcopy(feature.get("properties") or {})
        if not isinstance(props, dict):
            props = {}
        area = abs(GEOD.geometry_area_perimeter(geom)[0]) if geom.geom_type in ("Polygon", "MultiPolygon") else None
        length = GEOD.geometry_length(geom) if geom.geom_type in ("LineString", "MultiLineString") else None
        features.append({
            "id": f"CF-{i:04d}", "source_id": str(feature.get("id", ""))[:120],
            "geometry": mapping(geom), "geometry_type": geom.geom_type,
            "properties": props, "class": None, "class_note": "",
            "area_m2": round(area, 2) if area is not None else None,
            "length_m": round(length, 2) if length is not None else None,
        })
    return features, sha256(raw).hexdigest()


def assign_class(features: list[dict], feature_id: str, chosen: str, note: str = "") -> list[dict]:
    if chosen not in CLASSES:
        raise ValueError("Choose a supported class explicitly.")
    result = deepcopy(features)
    for f in result:
        if f["id"] == feature_id:
            f["class"] = chosen
            f["class_note"] = note.strip()[:500]
            return result
    raise ValueError("Feature was not found.")


def clear_class(features: list[dict], feature_id: str) -> list[dict]:
    result = deepcopy(features)
    for f in result:
        if f["id"] == feature_id:
            f["class"] = None
            f["class_note"] = ""
            return result
    raise ValueError("Feature was not found.")


def inventory(features: list[dict]) -> dict:
    areas = defaultdict(float)
    counts = Counter()
    non_area = Counter()
    classified = 0
    for f in features:
        cls = f.get("class")
        if cls:
            classified += 1
            counts[cls] += 1
            if f["area_m2"] is not None:
                areas[cls] += f["area_m2"]
            else:
                non_area[cls] += 1
    return {
        "features": len(features), "classified": classified,
        "unclassified": len(features) - classified,
        "completion_pct": 100 * classified / len(features) if features else 0,
        "areas_m2": {k: round(v, 2) for k, v in areas.items()},
        "counts": dict(counts), "non_area_features": dict(non_area),
    }


def export_geojson(features: list[dict]) -> bytes:
    output = {"type": "FeatureCollection", "features": []}
    for f in features:
        props = deepcopy(f["properties"])
        # Only explicit user assignment populates these fields.
        props.update({"cf_feature_id": f["id"], "cf_manual_class": f["class"],
                      "cf_class_note": f["class_note"], "cf_area_m2": f["area_m2"],
                      "cf_length_m": f["length_m"]})
        output["features"].append({"type": "Feature", "id": f["id"],
                                    "geometry": f["geometry"], "properties": props})
    return json.dumps(output, indent=2, ensure_ascii=False).encode("utf-8")
