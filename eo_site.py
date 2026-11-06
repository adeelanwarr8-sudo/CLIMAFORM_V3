"""Observed EO context for a user-defined WGS84 site; never master-plan predictions.

This module uses the legacy QA-aware satellite processors without changing their
scientific calculations. Scene discovery and raster processing are opt-in actions.
"""
from __future__ import annotations
import base64
import hashlib
import json

import folium
import streamlit as st
from streamlit_folium import st_folium
from shapely.geometry import shape
from pyproj import Geod
from buffer_geometry import create_analysis_buffer
from satellite_discovery import discover_landsat
from landsat_temperature import assess_temperature_candidate
from sentinel2_analysis import discover_sentinel, assess_sentinel

EO_KEYS = ("cf_eo_landsat", "cf_eo_lst", "cf_eo_sentinel", "cf_eo_indices")


def site_signature(boundary: dict | None, buffer_m: int) -> str | None:
    if boundary is None:
        return None
    payload = json.dumps({"geometry": boundary, "buffer_m": buffer_m}, sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()


def invalidate_site_results(state, signature):
    if state.get("cf_eo_site_signature") != signature:
        for key in EO_KEYS:
            state.pop(key, None)
        state["cf_eo_site_signature"] = signature
        state["cf_eo_map_revision"] = state.get("cf_eo_map_revision", 0) + 1
        return True
    return False


def _image_overlay(m, png, bounds, label, *, show=False):
    if png is None or bounds is None:
        return
    encoded = base64.b64encode(png).decode("ascii")
    folium.raster_layers.ImageOverlay(
        image="data:image/png;base64," + encoded,
        bounds=bounds, name=label, opacity=0.78,
        interactive=False, show=show,
    ).add_to(m)


def site_geometry_summary(boundary: dict, buffer_m: int) -> dict:
    """Geodesic area/perimeter of the original site; context buffer is separate."""
    geom = shape(boundary)
    if geom.geom_type != "Polygon" or geom.is_empty or not geom.is_valid:
        raise ValueError("Expected one valid site Polygon.")
    geod = Geod(ellps="WGS84")
    area, perimeter = geod.geometry_area_perimeter(geom)
    centroid = geom.centroid
    return {
        "area_m2": abs(float(area)),
        "perimeter_m": float(perimeter),
        "latitude": float(centroid.y),
        "longitude": float(centroid.x),
        "vertex_count": len(geom.exterior.coords) - 1,
        "buffer_m": int(buffer_m),
    }


def build_context_map(boundary, lst=None, indices=None, buffer_m: int = 25) -> folium.Map:
    geom = shape(boundary)
    centre = geom.centroid
    m = folium.Map(location=[centre.y, centre.x], zoom_start=16, tiles=None, control_scale=True)
    folium.TileLayer(tiles="OpenStreetMap", name="Street map", show=True).add_to(m)
    folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        name="Aerial imagery", attr="Tiles © Esri and contributors", show=False,
    ).add_to(m)
    context = create_analysis_buffer(boundary, buffer_m)
    if context is not None:
        folium.GeoJson(
            context.geometry,
            name=f"Analysis buffer · {buffer_m} m",
            style_function=lambda _: {
                "color": "#e58e28", "weight": 2.5,
                "dashArray": "7, 5", "fillColor": "#e58e28", "fillOpacity": 0.035,
            },
        ).add_to(m)
    folium.GeoJson(boundary, name="Geographical site boundary",
                   style_function=lambda _: {"color":"#08786d", "weight":3, "fillOpacity":0.07}).add_to(m)
    if lst is not None and lst.status == "SITE_QA_PASS":
        _image_overlay(m, lst.overlay_png, lst.overlay_bounds,
                       "Observed Landsat LST · " + lst.observed_utc[:10], show=True)
    if indices is not None and indices.status == "PRELIMINARY_QA_PASS":
        _image_overlay(m, indices.ndvi_overlay_png, indices.overlay_bounds,
                       "Observed Sentinel-2 NDVI · " + indices.observed_utc[:10])
        _image_overlay(m, indices.ndbi_overlay_png, indices.overlay_bounds,
                       "Observed Sentinel-2 NDBI · " + indices.observed_utc[:10])
    bounds = shape(context.geometry).bounds if context is not None else geom.bounds
    m.fit_bounds([[bounds[1],bounds[0]],[bounds[3],bounds[2]]],padding=(20,20))
    folium.LayerControl(collapsed=False).add_to(m)
    return m


def _metric_cards(lst, indices):
    a,b,c = st.columns(3)
    with a:
        valid = lst is not None and lst.status == "SITE_QA_PASS" and lst.mean_c is not None
        st.metric("Mean observed LST", f"{lst.mean_c:.1f} °C" if valid else "—")
        st.caption("Landsat · " + lst.observed_utc[:10] if valid else "Not yet assessed")
    with b:
        valid = indices is not None and indices.status == "PRELIMINARY_QA_PASS" and indices.ndvi_mean is not None
        st.metric("Mean NDVI", f"{indices.ndvi_mean:.3f}" if valid else "—")
        st.caption("Sentinel-2 · " + indices.observed_utc[:10] if valid else "Not yet assessed")
    with c:
        valid = indices is not None and indices.status == "PRELIMINARY_QA_PASS" and indices.ndbi_mean is not None
        st.metric("Mean NDBI", f"{indices.ndbi_mean:.3f}" if valid else "—")
        st.caption("Sentinel-2 · not a built-up fraction" if valid else "Not yet assessed")


def _site_info_panel(boundary: dict, buffer_m: int) -> None:
    values = site_geometry_summary(boundary, buffer_m)
    st.markdown("##### Site information")
    st.markdown("**Site area**")
    st.markdown(f"### {values['area_m2']:,.1f} m² ({values['area_m2'] / 10_000:,.3f} ha)")
    st.markdown("**Site perimeter**")
    st.write(f"{values['perimeter_m']:,.1f} m")
    st.markdown("**Centroid · WGS84**")
    st.write(f"{values['latitude']:.6f}, {values['longitude']:.6f}")
    st.markdown("**Boundary vertices**")
    st.write(str(values["vertex_count"]))
    st.divider()
    st.markdown("**Analysis buffer**")
    st.write(f"{buffer_m} m · context only")


def render_eo_site(boundary: dict | None) -> None:
    st.markdown("#### EO environmental intelligence")
    if boundary is None:
        invalidate_site_results(st.session_state, None)
        st.info("Define the geographical project boundary to enable the observed environmental snapshot.")
        return
    geom = shape(boundary)
    if geom.geom_type != "Polygon" or not geom.is_valid or geom.is_empty:
        st.warning("EO analysis requires one valid WGS84 Polygon site boundary.")
        return
    buffer_m = st.select_slider("EO context buffer (m)", options=[0,25,50,75,100,150,200,250,300],
                                value=25, key="cf_eo_buffer")
    signature = site_signature(boundary, buffer_m)
    invalidate_site_results(st.session_state, signature)
    lst = st.session_state.get("cf_eo_lst")
    indices = st.session_state.get("cf_eo_indices")
    st.markdown("##### Observed environmental indicators")
    st.caption("Values appear after satellite processing. Observations refer to the existing site, not a proposed master plan.")
    _metric_cards(lst, indices)
    map_col, site_col = st.columns([3.4, 1], gap="large")
    with map_col:
        st_folium(build_context_map(boundary, lst, indices, buffer_m=buffer_m), height=510,
                  key=f"cf_eo_map_{st.session_state.get('cf_eo_map_revision',0)}",
                  use_container_width=True, returned_objects=[])
        st.caption("Teal: geographical site boundary · Orange dashed: analysis-buffer boundary")
    with site_col:
        _site_info_panel(boundary, buffer_m)
    st.caption("Toggle observed LST, NDVI and NDBI independently using the map layer selector. These observations describe the physical site at their acquisition dates, not a proposed master plan.")
    with st.expander("Acquire observations & inspect quality", expanded=(lst is None and indices is None)):
        land, optic = st.columns(2)
        with land:
            st.markdown("**Landsat 8/9 · Surface temperature**")
            if st.button("Discover Landsat scene", key="cf_eo_landsat_find", use_container_width=True):
                try:
                    with st.spinner("Querying the Planetary Computer scene catalogue…"):
                        st.session_state["cf_eo_landsat"] = discover_landsat(boundary,buffer_m)
                    st.session_state.pop("cf_eo_lst",None)
                    st.rerun()
                except (ValueError, RuntimeError) as exc:
                    st.error(str(exc))
            found = st.session_state.get("cf_eo_landsat")
            if found is not None:
                if found.candidate is None:
                    st.warning("No qualifying Landsat surface-temperature scene within the permitted 3–45-day window.")
                else:
                    scene = found.candidate
                    st.caption(f"{scene.item_id} · {scene.observed_utc[:10]} · {scene.scene_cloud_pct:.1f}% scene cloud")
                    if st.button("Process observed LST & QA",key="cf_eo_lst_process",use_container_width=True):
                        try:
                            with st.spinner("Reading thermal / QA rasters…"):
                                value = assess_temperature_candidate(scene,boundary,buffer_m)
                            st.session_state["cf_eo_lst"] = value
                            st.session_state["cf_eo_map_revision"] += 1
                            st.rerun()
                        except (ValueError, RuntimeError, OSError) as exc:
                            st.error(str(exc))
                result = st.session_state.get("cf_eo_lst")
                if result is not None:
                    st.caption(f"Site QA: {result.status} · {result.valid_site_pixels}/{result.site_pixels} valid delivery pixels · {result.site_coverage_pct:.1f}% coverage")
                    for note in result.notes: st.caption(note)
        with optic:
            st.markdown("**Sentinel-2 · NDVI / NDBI**")
            if st.button("Discover Sentinel-2 scene",key="cf_eo_sentinel_find",use_container_width=True):
                try:
                    with st.spinner("Querying the Planetary Computer scene catalogue…"):
                        st.session_state["cf_eo_sentinel"] = discover_sentinel(boundary,buffer_m)
                    st.session_state.pop("cf_eo_indices",None)
                    st.rerun()
                except (ValueError, RuntimeError) as exc:
                    st.error(str(exc))
            found = st.session_state.get("cf_eo_sentinel")
            if found is not None:
                if found.candidate is None:
                    st.warning("No qualifying Sentinel-2 scene within the permitted 90/180-day search windows.")
                else:
                    scene = found.candidate
                    st.caption(f"{scene.item_id} · {scene.observed_utc[:10]} · {scene.scene_cloud_pct:.1f}% scene cloud")
                    if st.button("Process NDVI / NDBI & QA",key="cf_eo_indices_process",use_container_width=True):
                        try:
                            with st.spinner("Reading optical bands and SCL mask…"):
                                value = assess_sentinel(scene,boundary,buffer_m)
                            st.session_state["cf_eo_indices"] = value
                            st.session_state["cf_eo_map_revision"] += 1
                            st.rerun()
                        except (ValueError, RuntimeError, OSError) as exc:
                            st.error(str(exc))
                result=st.session_state.get("cf_eo_indices")
                if result is not None:
                    st.caption(f"Site QA: {result.status} · {result.valid_pixels}/{result.footprint_pixels} valid 20 m pixels · {result.valid_coverage_pct}% coverage")
                    for note in result.notes: st.caption(note)
    st.caption("LST: 30 m delivery grid, approximately 100 m native thermal information. NDVI/NDBI: 20 m grid. Single-scene observations; separate dates and cloud masks. No UHI intensity, air temperature or future design cooling prediction is calculated.")
