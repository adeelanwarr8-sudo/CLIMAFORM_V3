"""ClimaForma · DXF master-plan import, georeferencing and manual alignment.

One self-contained script. Use it in two ways:

  1. Standalone app:     streamlit run climaforma_dxf.py
  2. Inside ClimaForma:  from climaforma_dxf import dxf_placement_ui
                         placed = dxf_placement_ui(boundary_info.geometry)

Workflow
  Site boundary (drawn on the map or uploaded)
  -> upload DXF master plan
  -> read geometry per CAD layer (lines, polylines with arcs, circles, splines,
     hatches, blocks such as tree symbols)
  -> scale from the DXF units ($INSUNITS)  -- never fitted, never changed
  -> automatic rigid fit of the DXF site-boundary layer onto the drawn boundary
     (move + rotate, mirror check)
  -> user adjusts: nudge buttons, click-to-move, click-to-rotate about a pin
  -> Accept -> WGS84 GeoJSON FeatureCollection (property "layer" on every feature)
     plus alignment parameters to store with the project.

If the DXF is already in a real projected CRS (e.g. ETRS89 / UTM 31N,
EPSG:25831), the app detects the large coordinates and places it directly.

Requirements:
  pip install streamlit folium streamlit-folium ezdxf shapely>=2 pyproj numpy
"""

from __future__ import annotations


import io
from hashlib import sha256
import math
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone

import numpy as np
import ezdxf
from ezdxf import recover
from ezdxf import path as dxfpath
from pyproj import CRS, Transformer
from shapely import affinity
from shapely.geometry import (LineString, Point, Polygon, mapping, shape)
from shapely.ops import transform as shp_transform, unary_union

MAX_BYTES = 25 * 1024 * 1024
MAX_FEATURES = 25_000
FLATTEN_TOL_M = 0.05            # curve flattening tolerance (metres)
CLOSE_TOL_M = 0.05              # gap treated as closed (metres)

# $INSUNITS code -> metres per drawing unit
INSUNITS_TO_M = {1: 0.0254, 2: 0.3048, 3: 1609.344, 4: 0.001, 5: 0.01,
                 6: 1.0, 7: 1000.0, 8: 2.54e-8, 9: 2.54e-5, 10: 0.9144,
                 14: 0.1, 15: 10.0}
UNIT_NAMES = {0.0254: "inches", 0.3048: "feet", 0.001: "millimetres",
              0.01: "centimetres", 1.0: "metres", 0.1: "decimetres"}
PLAUSIBLE_SCALES = (1.0, 0.01, 0.001, 0.0254, 0.3048)


@dataclass
class DxfLayerSummary:
    name: str
    polygons: int = 0
    lines: int = 0
    points: int = 0


@dataclass
class DxfPlan:
    """Geometry read from a DXF, still in drawing units."""
    features: list[dict]                 # {"geom": shapely, "layer", "kind", "block"}
    layers: dict[str, DxfLayerSummary]
    unit_scale: float | None             # metres per unit from $INSUNITS
    extent: tuple[float, float, float, float]
    notes: list[str] = field(default_factory=list)

    def boundary_candidates(self, limit: int = 10) -> list[tuple[str, float]]:
        """Layers holding closed polygons, largest polygon first (for a picker)."""
        best: dict[str, float] = {}
        for f in self.features:
            if isinstance(f["geom"], Polygon):
                best[f["layer"]] = max(best.get(f["layer"], 0.0), f["geom"].area)
        return sorted(best.items(), key=lambda kv: -kv[1])[:limit]

    def looks_projected(self) -> bool:
        """Large coordinates suggest a national grid / UTM, not local CAD space."""
        x0, y0, x1, y1 = self.extent
        return abs(x0) > 100_000 and abs(y0) > 100_000 and (x1 - x0) < 50_000


@dataclass(frozen=True)
class Alignment:
    """Rigid placement of a DXF on the map. Scale is fixed by the units.

    Transform order (metres, site-centred Transverse Mercator):
      scale -> move pivot to origin -> mirror (optional) -> rotate -> move to offset
    The plan therefore rotates about its own pivot (the DXF boundary centroid),
    and offset_e/offset_n say where that pivot sits relative to the site centre.
    """
    lon0: float                      # projection origin = drawn site centroid
    lat0: float
    unit_scale: float                # metres per drawing unit (never fitted)
    pivot_x: float                   # pivot in drawing units
    pivot_y: float
    rotation_deg: float = 0.0        # counter-clockwise
    mirrored: bool = False
    offset_e: float = 0.0            # metres east of site centre
    offset_n: float = 0.0            # metres north of site centre

    def to_dict(self) -> dict:
        return {k: (round(v, 6) if isinstance(v, float) else v) for k, v in self.__dict__.items()}

    @classmethod
    def from_dict(cls, d: dict) -> "Alignment":
        return cls(**d)


@dataclass
class DxfPlacement:
    feature_collection: dict             # WGS84 GeoJSON of all DXF features
    dxf_boundary_wgs84: dict | None      # placed DXF site boundary (GeoJSON geometry)
    method: str                          # "boundary-fit" | "manual" | "projected-crs"
    alignment: Alignment | None
    iou: float | None = None             # overlap with the drawn boundary
    max_offset_m: float | None = None    # worst boundary mismatch
    processed_utc: str = ""
    notes: list[str] = field(default_factory=list)


# --------------------------------------------------------------------------
# 1. Reading
# --------------------------------------------------------------------------

def read_dxf(data: bytes) -> DxfPlan:
    if not data:
        raise ValueError("The DXF file is empty.")
    if len(data) > MAX_BYTES:
        raise ValueError(f"DXF is larger than {MAX_BYTES // 1024 // 1024} MB.")
    if data[:4] == b"AC10":
        raise ValueError("This looks like a DWG file. Please export it as DXF (Save As → DXF).")
    try:
        doc, auditor = recover.read(io.BytesIO(data))
    except (ezdxf.DXFStructureError, OSError) as exc:
        raise ValueError(f"Could not read DXF: {exc}") from exc

    notes: list[str] = []
    if auditor.has_errors:
        notes.append(f"DXF had {len(auditor.errors)} structural errors; recovered what was readable.")

    code = int(doc.header.get("$INSUNITS", 0) or 0)
    unit_scale = INSUNITS_TO_M.get(code)
    if unit_scale is None:
        notes.append("Drawing units are not set ($INSUNITS = unitless); scale will be estimated.")

    # Flatten curves to a tolerance that is ~5 cm in real units.
    tol = FLATTEN_TOL_M / (unit_scale or 0.001)
    close_tol = CLOSE_TOL_M / (unit_scale or 0.001)

    features: list[dict] = []
    layers: dict[str, DxfLayerSummary] = {}

    def add(geom, layer, kind, block=None):
        if geom is None or geom.is_empty:
            return
        if len(features) >= MAX_FEATURES:
            raise ValueError(f"DXF has more than {MAX_FEATURES:,} drawable entities. "
                             "Purge unused layers or export only the master-plan layers.")
        s = layers.setdefault(layer, DxfLayerSummary(layer))
        if isinstance(geom, Polygon):
            s.polygons += 1
        elif isinstance(geom, Point):
            s.points += 1
        else:
            s.lines += 1
        features.append({"geom": geom, "layer": layer, "kind": kind, "block": block})

    def handle(entity, layer_override=None, depth=0):
        etype = entity.dxftype()
        layer = layer_override or entity.dxf.get("layer", "0")
        if layer == "0" and layer_override:
            layer = layer_override
        if etype in ("TEXT", "MTEXT", "DIMENSION", "VIEWPORT", "IMAGE", "WIPEOUT",
                     "ATTDEF", "ATTRIB", "LEADER", "MLEADER"):
            return
        if etype == "INSERT":
            ins = entity.dxf.insert
            add(Point(ins.x, ins.y), layer, "block-insert", entity.dxf.name)
            # Keep one insertion point per block reference. Expanding the block
            # here would also produce constituent primitives and double-count
            # symbols such as tree circles. The user classifies the point.
            return
        if etype == "POINT":
            p = entity.dxf.location
            add(Point(p.x, p.y), layer, "point")
            return
        if etype == "HATCH":
            try:
                for p in dxfpath.from_hatch(entity):
                    _add_path(p, layer, "hatch", add, tol, close_tol, force_polygon=True)
            except Exception:
                pass
            return
        try:
            p = dxfpath.make_path(entity)
        except (TypeError, ValueError):
            return  # unsupported entity type (3D solids, etc.)
        _add_path(p, layer, etype.lower(), add, tol, close_tol)

    msp = doc.modelspace()
    for e in msp:
        handle(e)

    if not features:
        raise ValueError("No drawable geometry found in the DXF model space.")

    bounds = unary_union([f["geom"] for f in features]).bounds
    return DxfPlan(features, layers, unit_scale, bounds, notes)


def _add_path(p, layer, kind, add, tol, close_tol, force_polygon=False):
    pts = [(v.x, v.y) for v in p.flattening(distance=tol)]
    # drop consecutive duplicates
    pts = [pt for i, pt in enumerate(pts) if i == 0 or pt != pts[i - 1]]
    if len(pts) < 2:
        return
    gap = math.dist(pts[0], pts[-1])
    closed = p.is_closed or gap <= close_tol or force_polygon
    if closed and len(pts) >= 3:
        poly = Polygon(pts)
        if not poly.is_valid:
            poly = poly.buffer(0)
        if poly.area > 0:
            if poly.geom_type == "MultiPolygon":
                for part in poly.geoms:
                    add(part, layer, kind)
            else:
                add(poly, layer, kind)
            return
    add(LineString(pts), layer, kind)


# --------------------------------------------------------------------------
# 2. Placement on the map
# --------------------------------------------------------------------------

def site_crs(lon0: float, lat0: float) -> CRS:
    """Transverse Mercator centred on the site: scale error is effectively zero."""
    return CRS.from_proj4(f"+proj=tmerc +lat_0={lat0} +lon_0={lon0} +k=1 "
                          "+x_0=0 +y_0=0 +ellps=WGS84 +units=m +no_defs")


def _transformers(al: Alignment):
    crs = site_crs(al.lon0, al.lat0)
    return (Transformer.from_crs(4326, crs, always_xy=True),
            Transformer.from_crs(crs, 4326, always_xy=True))


def _rigid(al: Alignment):
    """Return a function mapping drawing-unit geometry to site metres."""
    s = al.unit_scale
    def apply(g):
        g = affinity.translate(g, -al.pivot_x, -al.pivot_y)
        g = affinity.scale(g, s, s, origin=(0, 0))
        if al.mirrored:
            g = affinity.scale(g, -1, 1, origin=(0, 0))
        g = affinity.rotate(g, al.rotation_deg, origin=(0, 0))
        return affinity.translate(g, al.offset_e, al.offset_n)
    return apply


def site_boundary_layer_geometry(plan: DxfPlan, boundary_layer: str | None) -> Polygon:
    polys = [f["geom"] for f in plan.features if isinstance(f["geom"], Polygon)
             and (boundary_layer is None or f["layer"] == boundary_layer)]
    if not polys:
        raise ValueError("No closed polygon found on the selected boundary layer. "
                         "Close the site-boundary polyline in CAD or pick another layer.")
    return max(polys, key=lambda g: g.area)


def initial_alignment(plan: DxfPlan, site_geojson: dict,
                      boundary_layer: str | None = None) -> tuple[Alignment, list[str]]:
    """Units + centre-on-site, no rotation. Starting point for fitting or manual work."""
    site_ll = shape(site_geojson)
    c = site_ll.centroid
    notes: list[str] = []
    try:
        ref = site_boundary_layer_geometry(plan, boundary_layer)
    except ValueError:
        if boundary_layer is not None:
            raise
        ref = unary_union([f["geom"] for f in plan.features]).convex_hull
        notes.append("No closed boundary polygon found; centred the whole drawing on the site.")
    al0 = Alignment(c.x, c.y, plan.unit_scale or 1.0, ref.centroid.x, ref.centroid.y)
    fwd, _ = _transformers(al0)
    site_m = shp_transform(fwd.transform, site_ll)

    scale = plan.unit_scale
    if scale is None:
        raw = math.sqrt(site_m.area / ref.area)
        scale = min(PLAUSIBLE_SCALES, key=lambda s: abs(math.log(raw / s)))
        notes.append(f"Units estimated as {UNIT_NAMES.get(scale, scale)} from the boundary "
                     "area; please confirm.")
    else:
        ratio = site_m.area / (ref.area * scale ** 2)
        if not 0.65 < ratio < 1.5:
            notes.append(f"Warning: the drawn site is {ratio:.2f}× the DXF boundary area. "
                         "Check the boundary layer and the drawing units.")
    sc = site_m.centroid
    return replace(al0, unit_scale=scale, offset_e=sc.x, offset_n=sc.y), notes


def fit_to_boundary(plan: DxfPlan, site_geojson: dict, boundary_layer: str | None = None,
                    allow_mirror: bool = True) -> tuple[Alignment, list[str]]:
    """Automatic rigid fit: best rotation (and mirror) of the DXF boundary onto the drawn one."""
    al, notes = initial_alignment(plan, site_geojson, boundary_layer)
    boundary = site_boundary_layer_geometry(plan, boundary_layer)
    fwd, _ = _transformers(al)
    site_m = shp_transform(fwd.transform, shape(site_geojson))

    def iou(a: Alignment, g) -> float:
        p = _rigid(a)(g)
        u = p.union(site_m).area
        return p.intersection(site_m).area / u if u else 0.0

    coarse = boundary.simplify(0.2 / al.unit_scale)
    best = (-1.0, al)
    for mirror in ((False, True) if allow_mirror else (False,)):
        for ang in np.arange(0.0, 360.0, 2.0):
            cand = replace(al, mirrored=mirror, rotation_deg=float(ang))
            v = iou(cand, coarse)
            if v > best[0]:
                best = (v, cand)
    for step in (0.5, 0.1, 0.02):
        a0 = best[1].rotation_deg
        for ang in np.arange(a0 - 4 * step, a0 + 4 * step + 1e-9, step):
            cand = replace(best[1], rotation_deg=float(ang) % 360)
            v = iou(cand, boundary)
            if v > best[0]:
                best = (v, cand)
    if best[1].mirrored:
        notes.append("Best fit required mirroring the drawing; confirm it is not flipped in CAD.")
    if best[0] < 0.85:
        notes.append(f"Low automatic fit (IoU {best[0]:.2f}). Adjust manually using the "
                     "aerial basemap.")
    return best[1], notes


def nudge(al: Alignment, rotate_deg: float = 0.0, east_m: float = 0.0,
          north_m: float = 0.0) -> Alignment:
    """Manual adjustment. Rotation is about the plan's own pivot; scale never changes."""
    return replace(al, rotation_deg=(al.rotation_deg + rotate_deg) % 360,
                   offset_e=al.offset_e + east_m, offset_n=al.offset_n + north_m)


def move_by_points(al: Alignment, from_lonlat: tuple[float, float],
                   to_lonlat: tuple[float, float]) -> Alignment:
    """Translate so the plan point clicked first lands on the map point clicked second."""
    fwd, _ = _transformers(al)
    x0, y0 = fwd.transform(*from_lonlat)
    x1, y1 = fwd.transform(*to_lonlat)
    return nudge(al, east_m=x1 - x0, north_m=y1 - y0)


def rotate_by_points(al: Alignment, from_lonlat: tuple[float, float],
                     to_lonlat: tuple[float, float],
                     anchor_lonlat: tuple[float, float] | None = None) -> Alignment:
    """Rotate so the clicked plan point turns toward the target point.

    With `anchor_lonlat` (e.g. the point just placed with move_by_points) the plan
    turns about that anchor, so the anchor stays put: classic two-point alignment.
    Without it the plan turns about its own pivot.
    """
    fwd, _ = _transformers(al)
    ax, ay = fwd.transform(*anchor_lonlat) if anchor_lonlat else (al.offset_e, al.offset_n)
    x0, y0 = fwd.transform(*from_lonlat)
    x1, y1 = fwd.transform(*to_lonlat)
    theta = math.atan2(y1 - ay, x1 - ax) - math.atan2(y0 - ay, x0 - ax)
    c, s = math.cos(theta), math.sin(theta)
    dx_, dy_ = al.offset_e - ax, al.offset_n - ay
    return replace(al, rotation_deg=(al.rotation_deg + math.degrees(theta)) % 360,
                   offset_e=ax + c * dx_ - s * dy_, offset_n=ay + s * dx_ + c * dy_)


def place(plan: DxfPlan, al: Alignment, site_geojson: dict | None = None,
          boundary_layer: str | None = None, method: str = "manual",
          notes: list[str] | None = None) -> DxfPlacement:
    """Apply an alignment and return map-ready WGS84 GeoJSON plus fit diagnostics."""
    _, inv = _transformers(al)
    to_ll = lambda x, y, z=None: inv.transform(x, y)
    apply = _rigid(al)
    placed = [{**f, "geom": apply(f["geom"])} for f in plan.features]
    fc = _to_feature_collection(placed, to_ll)

    boundary_ll, iou, max_off = None, None, None
    try:
        b = apply(site_boundary_layer_geometry(plan, boundary_layer))
        boundary_ll = _round(mapping(shp_transform(to_ll, b)))
        if site_geojson is not None:
            fwd, _ = _transformers(al)
            site_m = shp_transform(fwd.transform, shape(site_geojson))
            iou = round(b.intersection(site_m).area / b.union(site_m).area, 4)
            max_off = round(b.exterior.hausdorff_distance(site_m.exterior), 2)
    except ValueError:
        pass
    return DxfPlacement(fc, boundary_ll, method, al, iou, max_off, _now(), list(notes or []))


def place_by_crs(plan: DxfPlan, epsg: int, site_geojson: dict | None = None) -> DxfPlacement:
    """Drawing already in a projected CRS (e.g. EPSG:25831 for Barcelona). No fitting."""
    # Projected X/Y coordinates are already expressed in their declared CRS
    # units. Multiplying entire eastings/northings by $INSUNITS can displace
    # a valid survey by hundreds of kilometres.
    declared = CRS.from_epsg(epsg)
    if not declared.is_projected:
        raise ValueError("Choose a projected CRS for DXF coordinate placement.")
    to_wgs = Transformer.from_crs(declared, 4326, always_xy=True)
    fn = lambda x, y, z=None: to_wgs.transform(x, y)
    fc = _to_feature_collection(plan.features, fn)
    notes = [f"Placed using EPSG:{epsg}; coordinate magnitudes were not rescaled."]
    if plan.unit_scale not in (None, 1.0):
        notes.append("DXF $INSUNITS differs from metre-based projected coordinates. "
                     "Verify the survey CRS and exported units before accepting.")
    if site_geojson is not None:
        plan_ll = unary_union([shp_transform(fn, f["geom"]) for f in plan.features])
        if not plan_ll.envelope.intersects(shape(site_geojson)):
            notes.append("Warning: the drawing does not overlap the site boundary. "
                         "Check the CRS or use boundary fitting instead.")
    return DxfPlacement(fc, None, "projected-crs", None, processed_utc=_now(), notes=notes)


def _to_feature_collection(features, fn) -> dict:
    out = []
    for f in features:
        g = shp_transform(fn, f["geom"])
        out.append({"type": "Feature",
                    "properties": {"layer": f["layer"], "kind": f["kind"],
                                   "block": f["block"] or ""},
                    "geometry": _round(mapping(g))})
    return {"type": "FeatureCollection", "features": out}


def _round(geom: dict, nd: int = 8) -> dict:
    def r(c):
        return [r(x) for x in c] if isinstance(c[0], (list, tuple)) else [round(c[0], nd), round(c[1], nd)]
    return {"type": geom["type"], "coordinates": r(geom["coordinates"])}


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def layer_color(name: str) -> str:
    """Stable, readable colour per layer for map display."""
    palette = ["#2f7d4f", "#b5651d", "#3b6ea8", "#8e44ad", "#c0392b",
               "#16a085", "#7f8c2d", "#d35400", "#5d6d7e", "#a93226"]
    return palette[sum(map(ord, name)) % len(palette)]


# ==========================================================================
# 3. Streamlit user interface
# ==========================================================================
# Streamlit and folium are imported here so the geometry functions above can
# be used (and tested) without a UI.
import json as _json

import folium
import streamlit as st
from folium.plugins import Draw
from streamlit_folium import st_folium

ESRI_IMAGERY = ("https://server.arcgisonline.com/ArcGIS/rest/services/"
                "World_Imagery/MapServer/tile/{z}/{y}/{x}")
DEFAULT_CENTER = (41.3874, 2.1686)          # Barcelona


def _base_map(lat: float, lon: float, zoom: int) -> folium.Map:
    m = folium.Map(location=[lat, lon], zoom_start=zoom, tiles=None, max_zoom=22,
                   control_scale=True)
    folium.TileLayer("OpenStreetMap", name="Street map", show=True, max_zoom=22,
                     max_native_zoom=19).add_to(m)
    folium.TileLayer(ESRI_IMAGERY, name="Aerial imagery", attr="Tiles © Esri and contributors",
                     show=False, max_zoom=22, max_native_zoom=19).add_to(m)
    return m


def _geometry_from_geojson(data: dict) -> dict:
    if data.get("type") == "FeatureCollection":
        data = data["features"][0]
    if data.get("type") == "Feature":
        data = data["geometry"]
    if data.get("type") not in ("Polygon", "MultiPolygon"):
        raise ValueError("The site boundary must be a polygon.")
    return data


def site_boundary_ui() -> dict | None:
    """Step 1 (standalone only): draw or upload the site boundary."""
    site = st.session_state.get("site_geojson")
    if site is not None:
        st.sidebar.success("Site boundary defined.")
        if st.sidebar.button("Redraw site boundary", use_container_width=True):
            for k in list(st.session_state):
                if k.startswith("dxf_") or k == "site_geojson":
                    st.session_state.pop(k, None)
            st.rerun()
        return site

    st.subheader("1 · Define the site boundary")
    st.caption("Draw one polygon or rectangle around the site on the aerial image, "
               "or upload a GeoJSON polygon in the sidebar.")
    up = st.sidebar.file_uploader("Site boundary GeoJSON (optional)", type=["geojson", "json"])
    if up is not None:
        try:
            st.session_state["site_geojson"] = _geometry_from_geojson(_json.loads(up.getvalue()))
            st.rerun()
        except (ValueError, KeyError, IndexError, _json.JSONDecodeError) as exc:
            st.sidebar.error(f"Could not use this GeoJSON: {exc}")

    m = _base_map(*DEFAULT_CENTER, 15)
    Draw(export=False, position="topleft",
         draw_options={"polyline": False, "circle": False, "circlemarker": False,
                       "marker": False, "polygon": {"allowIntersection": False},
                       "rectangle": True},
         edit_options={"edit": {"selectedPathOptions": {"maintainColor": True,
                                                        "dashArray": "6, 6"}},
                       "remove": True}).add_to(m)
    folium.LayerControl(collapsed=True).add_to(m)
    out = st_folium(m, key="site_draw_map", height=560, use_container_width=True,
                    returned_objects=["all_drawings"])
    drawings = (out or {}).get("all_drawings") or []
    polys = [d["geometry"] for d in drawings
             if d.get("geometry", {}).get("type") == "Polygon"]
    if len(polys) > 1:
        st.warning("Draw only one site boundary; delete the extra shapes.")
    elif polys and st.button("Use this site boundary", type="primary"):
        st.session_state["site_geojson"] = polys[0]
        st.rerun()
    return None


def dxf_placement_ui(site: dict, key: str = "dxf") -> DxfPlacement | None:
    """Upload, place and adjust a DXF master plan on `site` (GeoJSON polygon, WGS84).

    Returns the accepted DxfPlacement (or None until the user accepts).
    All state lives in st.session_state under keys starting with `key`.
    """
    K = lambda name: f"{key}_{name}"
    st.subheader("2 · Place the master plan (DXF)")
    dxf_up = st.sidebar.file_uploader("Master plan DXF", type=["dxf"], key=K("upload"))
    if dxf_up is None:
        st.info("Upload the landscape / master plan as DXF (in CAD: Save As → DXF). "
                "It will be placed on the site boundary automatically.")
        return None

    # ---- read once per file + site --------------------------------------
    file_id = (sha256(dxf_up.getvalue()).hexdigest(), _json.dumps(site, sort_keys=True))
    if st.session_state.get(K("file_id")) != file_id:
        try:
            with st.spinner("Reading DXF…"):
                st.session_state[K("plan")] = read_dxf(dxf_up.getvalue())
        except ValueError as exc:
            st.error(str(exc))
            return None
        st.session_state[K("file_id")] = file_id
        for n in ("alignment", "auto", "notes", "click_first", "last_click",
                  "accepted", "anchor", "method", "confirm_units", "use_crs",
                  "layers", "boundary_layer", "epsg"):

            st.session_state.pop(K(n), None)
    plan: DxfPlan = st.session_state[K("plan")]

    # ---- sidebar: layers, CRS, initial placement --------------------------
    st.sidebar.caption("Drawing units: "
                       + (UNIT_NAMES.get(plan.unit_scale, f"{plan.unit_scale:g} m")
                          if plan.unit_scale else "not set in file")
                       + f" · {len(plan.features):,} features · {len(plan.layers)} layers")
    for n in plan.notes:
        st.sidebar.caption(n)

    confirmed_units = st.sidebar.checkbox(
        "I have verified the drawing units and coordinate reference",
        value=False, key=K("confirm_units"),
        help="DXF $INSUNITS can be absent or incorrect. Confirm against known lengths in the drawing.")

    if plan.looks_projected():
        st.sidebar.info("Coordinates look like a real map grid (e.g. UTM).")
        use_crs = st.sidebar.checkbox("Place using a CRS (no fitting)", value=True, key=K("use_crs"))
    else:
        use_crs = False

    all_layers = sorted(plan.layers)
    shown = st.sidebar.multiselect("Layers to show", all_layers, default=all_layers,
                                   key=K("layers"))

    if use_crs:
        epsg = int(st.sidebar.number_input("Drawing CRS (EPSG code)", value=25831, step=1,
                                           key=K("epsg"),
                                           help="25831 = ETRS89 / UTM 31N (Barcelona); "
                                                "3006 = SWEREF99 TM (Sweden)."))
        try:
            placed = place_by_crs(plan, epsg, site)
        except Exception as exc:  # invalid EPSG etc.
            st.error(f"Could not place with EPSG:{epsg}: {exc}")
            return None
        for n in placed.notes:
            st.warning(n)
        _show_map(site, placed, shown, None, None, K)
        if st.button("Accept placement", type="primary", key=K("accept_crs"),
                     disabled=not confirmed_units):
            st.session_state[K("accepted")] = placed
        if st.session_state.get(K("accepted")):
            st.success("Placement accepted.")
        return st.session_state.get(K("accepted"))

    candidates = [name for name, _ in plan.boundary_candidates()]
    b_layer = st.sidebar.selectbox("DXF layer with the site boundary", candidates or ["(none)"],
                                   key=K("boundary_layer"),
                                   help="Layers with the largest closed polygons first.")
    b_layer = None if b_layer == "(none)" else b_layer

    c_fit, c_centre = st.sidebar.columns(2)
    if (c_fit.button("Auto-fit", use_container_width=True, key=K("autofit"))
            or K("alignment") not in st.session_state):
        try:
            al, notes = fit_to_boundary(plan, site, b_layer)
        except ValueError:
            al, notes = initial_alignment(plan, site, None)
            notes.append("Automatic fit needs a closed boundary polygon; "
                         "placed at the site centre — align manually.")
        st.session_state.update({K("alignment"): al, K("auto"): al, K("notes"): notes})
        for n in ("accepted", "anchor", "click_first"):
            st.session_state.pop(K(n), None)
    if c_centre.button("Centre only", use_container_width=True, key=K("centre"),
                       help="Units and position only, no rotation: align manually."):
        try:
            al, notes = initial_alignment(plan, site, b_layer)
        except ValueError as exc:
            st.sidebar.error(str(exc))
        else:
            st.session_state.update({K("alignment"): al, K("notes"): notes})
            for n in ("accepted", "anchor", "click_first"):
                st.session_state.pop(K(n), None)

    al: Alignment = st.session_state[K("alignment")]
    placed = place(plan, al, site, b_layer, method="manual",
                   notes=st.session_state.get(K("notes")))

    map_col, ctl_col = st.columns([2.3, 1], gap="large")

    # ---- adjustment controls --------------------------------------------
    with ctl_col:
        st.markdown("**Adjust placement**")
        st.caption("Moves and rotations are rigid: lengths and areas stay exactly as drawn.")
        mstep = st.select_slider("Move step (m)", [0.1, 0.25, 0.5, 1.0, 2.0, 5.0],
                                 value=1.0, key=K("mstep"))
        rstep = st.select_slider("Rotate step (°)", [0.1, 0.25, 0.5, 1.0, 5.0, 15.0],
                                 value=1.0, key=K("rstep"))

        def do(**kw):
            st.session_state[K("alignment")] = nudge(st.session_state[K("alignment")], **kw)
            st.session_state.pop(K("accepted"), None)

        _, c_n, _ = st.columns(3)
        c_n.button("↑ N", on_click=do, kwargs={"north_m": mstep},
                   use_container_width=True, key=K("n"))
        c_w, _, c_e = st.columns(3)
        c_w.button("← W", on_click=do, kwargs={"east_m": -mstep},
                   use_container_width=True, key=K("w"))
        c_e.button("E →", on_click=do, kwargs={"east_m": mstep},
                   use_container_width=True, key=K("e"))
        _, c_s, _ = st.columns(3)
        c_s.button("↓ S", on_click=do, kwargs={"north_m": -mstep},
                   use_container_width=True, key=K("s"))
        c_rl, c_rr = st.columns(2)
        c_rl.button("⟲ Rotate left", on_click=do, kwargs={"rotate_deg": rstep},
                    use_container_width=True, key=K("rl"))
        c_rr.button("⟳ Rotate right", on_click=do, kwargs={"rotate_deg": -rstep},
                    use_container_width=True, key=K("rr"))

        st.divider()
        click_mode = st.radio("Map click", ["Off", "Move: plan point → target",
                                            "Rotate: plan point → target direction"],
                              key=K("click_mode"),
                              help="Click a recognisable point of the plan (e.g. a building "
                                   "corner), then where it really is on the aerial image.")
        first = st.session_state.get(K("click_first"))
        if st.session_state.get(K("anchor")):
            st.caption("Blue dot = pinned point from your last move; "
                       "click-rotation turns about it.")
            if st.button("Unpin", key=K("unpin")):
                st.session_state.pop(K("anchor"), None)
                st.rerun()
        if click_mode != "Off":
            st.info("Now click the target location on the map." if first
                    else "Click a point of the plan on the map.")

        st.divider()
        auto = st.session_state.get(K("auto"))
        st.markdown(f"**Rotation** {al.rotation_deg:.2f}°"
                    + (f" · {(al.rotation_deg - auto.rotation_deg + 180) % 360 - 180:+.2f}° "
                       "vs auto-fit" if auto else ""))
        if auto:
            st.markdown(f"**Moved** {al.offset_e - auto.offset_e:+.2f} m E · "
                        f"{al.offset_n - auto.offset_n:+.2f} m N vs auto-fit")
        if placed.iou is not None:
            st.markdown(f"**Match with drawn boundary** IoU {placed.iou:.2f} · "
                        f"max gap {placed.max_offset_m:.1f} m")
        st.caption(f"Scale fixed: 1 drawing unit = {al.unit_scale:g} m"
                   + (" · mirrored" if al.mirrored else ""))
        for n in placed.notes:
            st.warning(n)

        c_reset, c_ok = st.columns(2)
        if c_reset.button("Reset to auto-fit", use_container_width=True,
                          disabled=auto is None, key=K("reset")):
            st.session_state[K("alignment")] = auto
            for n in ("accepted", "anchor", "click_first"):
                st.session_state.pop(K(n), None)
            st.rerun()
        if c_ok.button("Accept placement", type="primary", use_container_width=True,
                       disabled=not confirmed_units, key=K("accept")):
            st.session_state[K("accepted")] = placed

    # ---- map --------------------------------------------------------------
    with map_col:
        out = _show_map(site, placed, shown, st.session_state.get(K("anchor")), first, K)
        click = (out or {}).get("last_clicked")
        if click_mode != "Off" and click and click != st.session_state.get(K("last_click")):
            st.session_state[K("last_click")] = click
            pt = (click["lng"], click["lat"])
            if first is None:
                st.session_state[K("click_first")] = pt
            else:
                if click_mode.startswith("Move"):
                    st.session_state[K("alignment")] = move_by_points(al, first, pt)
                    st.session_state[K("anchor")] = pt
                else:
                    st.session_state[K("alignment")] = rotate_by_points(
                        al, first, pt, st.session_state.get(K("anchor")))
                st.session_state.pop(K("click_first"), None)
                st.session_state.pop(K("accepted"), None)
            st.rerun()
        elif click_mode == "Off" and first:
            st.session_state.pop(K("click_first"), None)

    accepted = st.session_state.get(K("accepted"))
    if accepted is not None:
        st.success("Placement accepted.")
        c1, c2 = st.columns(2)
        c1.download_button("Download placed plan (GeoJSON)",
                           _json.dumps(accepted.feature_collection),
                           file_name="masterplan_wgs84.geojson",
                           mime="application/geo+json", key=K("dl_geojson"))
        if accepted.alignment is not None:
            c2.download_button("Download alignment parameters (JSON)",
                               _json.dumps(accepted.alignment.to_dict(), indent=2),
                               file_name="masterplan_alignment.json",
                               mime="application/json", key=K("dl_params"))
            with st.expander("Alignment parameters (store with the project)"):
                st.json(accepted.alignment.to_dict())
    return accepted


def _show_map(site, placed, shown, anchor, first, K):
    """Constant base map + one changing feature group, so zoom/pan survive reruns."""
    c = shape(site).centroid
    m = _base_map(c.y, c.x, 18)
    folium.GeoJson(site, name="Drawn site boundary",
                   style_function=lambda _: {"color": "#00e0c0", "weight": 3,
                                             "fillOpacity": 0, "dashArray": "6 5"}).add_to(m)
    folium.LayerControl(collapsed=True).add_to(m)

    fg = folium.FeatureGroup(name="DXF master plan")
    feats = [f for f in placed.feature_collection["features"]
             if f["properties"]["layer"] in shown]
    if feats:
        folium.GeoJson({"type": "FeatureCollection", "features": feats},
                       style_function=lambda f: {"color": layer_color(f["properties"]["layer"]),
                                                 "weight": 1.5, "fillOpacity": 0.25,
                                                 "fillColor": layer_color(f["properties"]["layer"])},
                       tooltip=folium.GeoJsonTooltip(["layer"])).add_to(fg)
    if placed.dxf_boundary_wgs84:
        folium.GeoJson(placed.dxf_boundary_wgs84,
                       style_function=lambda _: {"color": "#ff6b00", "weight": 3,
                                                 "fillOpacity": 0}).add_to(fg)
    if anchor:
        folium.CircleMarker([anchor[1], anchor[0]], radius=7, color="#ffffff", weight=2,
                            fill=True, fill_color="#0077ff", fill_opacity=1,
                            tooltip="Pinned point: rotation turns about here").add_to(fg)
    if first:
        folium.CircleMarker([first[1], first[0]], radius=6, color="#ff006e",
                            fill=True, fill_opacity=1).add_to(fg)

    out = st_folium(m, feature_group_to_add=fg, key=K("map"), height=620,
                    use_container_width=True, returned_objects=["last_clicked"])
    st.caption("Cyan dashed: drawn site boundary · Orange: DXF site boundary · "
               "Align to real features on the aerial image (kerbs, building corners).")
    return out


def add_placement_to_map(m: folium.Map, placed: DxfPlacement,
                         layers: list[str] | None = None) -> None:
    """Helper for ClimaForma's own maps: draw an accepted placement on any folium map."""
    feats = [f for f in placed.feature_collection["features"]
             if (layers is None or f["properties"]["layer"] in layers)
             and f["properties"]["kind"] != "block-insert"]
    if feats:
        folium.GeoJson({"type": "FeatureCollection", "features": feats},
                       name="Master plan (DXF)",
                       style_function=lambda f: {"color": layer_color(f["properties"]["layer"]),
                                                 "weight": 1.5, "fillOpacity": 0.25,
                                                 "fillColor": layer_color(f["properties"]["layer"])},
                       tooltip=folium.GeoJsonTooltip(["layer"])).add_to(m)


def main() -> None:
    st.set_page_config(page_title="ClimaForma · DXF master plan", layout="wide")
    st.title("Place a DXF master plan on the site")
    site = site_boundary_ui()
    if site is not None:
        dxf_placement_ui(site)


if __name__ == "__main__":      # true under `streamlit run climaforma_dxf.py`
    main()
