"""ClimaForma v3.0 — Master Plan Intelligence | UI-first prototype.

No EO model, UHI score, NDVI change, or future temperature is invented here.
The prior ClimaForma v2 repository should remain untouched during evaluation.
"""
import html
import re
import json

import folium
import pandas as pd
import streamlit as st
from streamlit_folium import st_folium
from masterplan import CLASSES, PALETTE, parse_geojson, assign_class, clear_class, inventory, export_geojson
from shapely.geometry import shape
from hashlib import sha256
from climaforma_dxf import site_boundary_ui, dxf_placement_ui

st.set_page_config(page_title="ClimaForma | Master Plan Intelligence", page_icon="◈", layout="wide", initial_sidebar_state="expanded")
st.markdown("""<style>
:root{--brand:#137b71;--dark:#173936;--muted:#667f7b}
.stApp{background:#f7f9f7;color:#203c39}
.block-container{max-width:1550px;padding-top:1.5rem;padding-bottom:4rem}
[data-testid="stSidebar"]{background:#143c39;color:#eaf7f1}
[data-testid="stSidebar"] *{color:inherit}
[data-testid="stSidebar"] .stButton button{color:#163e39;background:#e2f3e8;border:0}
[data-testid="stSidebar"] .stCaption{color:#c1d6d0}
h1,h2,h3{letter-spacing:-.035em}
.cf-logo{font-size:1.75rem;font-weight:850;color:#f2f8f1;letter-spacing:-.06em}
.cf-tag{color:#b5d5c9;font-size:.79rem;letter-spacing:.04em}
.cf-kicker{font-size:.73rem;font-weight:800;color:#08796c;letter-spacing:.14em;text-transform:uppercase}
.cf-title{font-size:2.15rem;font-weight:820;color:#183d38;letter-spacing:-.05em;line-height:1.18}
.cf-intro{font-size:1.02rem;color:#657a77;line-height:1.55;max-width:980px}
.cf-panel{background:#fff;border:1px solid #dfeae5;border-radius:13px;padding:17px 19px;margin:9px 0 17px}
.cf-label{font-size:.73rem;font-weight:850;letter-spacing:.1em;color:#2d8276;text-transform:uppercase}
.cf-value{font-size:1.45rem;color:#23453f;font-weight:800}
.cf-note{font-size:.79rem;line-height:1.5;color:#677f7a}
[data-testid="stTabs"] button{font-weight:680;font-size:.88rem}
[data-testid="stTabs"] button[aria-selected="true"]{color:#12786d}
</style>""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown('<div class="cf-logo">◈ ClimaForma</div><div class="cf-tag">MASTER PLAN INTELLIGENCE · v3.0 UI</div>', unsafe_allow_html=True)
    st.divider()
    st.markdown("**PROJECT**")
    project_name = st.text_input("Project name", value="New climate-responsive master plan", label_visibility="collapsed", max_chars=90)
    st.caption("Session-only prototype · No data is sent to an EO service")
    st.divider()
    st.markdown("**WORKFLOW**")
    st.markdown("01 · Project & site  \n02 · Master plan classification  \n03 · Baseline assessment  \n04 · Design options  \n05 · Compare performance")
    st.divider()
    st.markdown("**SCIENTIFIC STATUS**")
    st.caption("Geometry and manual classification are live. EO retrieval, thermal simulation, UHI and future NDVI prediction are not included in this UI prototype.")
    st.divider()
    st.caption("The established ClimaForma v2 application remains separate and unchanged.")

st.markdown('<div class="cf-kicker">Earth observation × architectural design</div>', unsafe_allow_html=True)
st.markdown('<div class="cf-title">Evaluate the plan. Improve the climate.</div>', unsafe_allow_html=True)
st.markdown('<div class="cf-intro">Upload a GeoJSON or DXF master plan, identify each design element yourself, and prepare a transparent baseline for microclimatic comparison. No automatic land-cover classification.</div>', unsafe_allow_html=True)
st.write("")

if "plan_features" not in st.session_state:
    st.session_state.plan_features = []
if "plan_sha" not in st.session_state:
    st.session_state.plan_sha = None
if "selected_feature" not in st.session_state:
    st.session_state.selected_feature = None

summary = inventory(st.session_state.plan_features)
summary_strip = st.columns(4)
for column, title, val, hint in zip(summary_strip,
    ("Imported geometries", "Manually classified", "Unclassified", "Classification progress"),
    (str(summary["features"]), str(summary["classified"]), str(summary["unclassified"]), f'{summary["completion_pct"]:.0f}%'),
    ("GeoJSON / DXF features", "Explicit assignments", "Requires user review", "Features, not site coverage")):
    with column:
        st.markdown(f'<div class="cf-panel"><div class="cf-label">{title}</div><div class="cf-value">{val}</div><div class="cf-note">{hint}</div></div>', unsafe_allow_html=True)

site_tab, classify_tab, baseline_tab, options_tab, compare_tab = st.tabs([
    "01 · Project & Site", "02 · Classify Master Plan", "03 · Baseline Assessment",
    "04 · Design Options", "05 · Compare Performance"
])

with site_tab:
    st.subheader("Start with your master plan · GeoJSON or DXF")
    st.caption("Choose GeoJSON (WGS84) or DXF (CAD geometry with an explicitly verified position). All design classes are assigned manually after import.")
    input_format = st.radio("Master plan input", ["GeoJSON", "DXF"], horizontal=True, key="master_plan_input_format")
    left, right = st.columns([1, 1], gap="large")
    if input_format == "GeoJSON":
        with left:
            upload = st.file_uploader("Upload master-plan GeoJSON", type=["geojson", "json"],
                                      key="masterplan_file", help="Up to 5 MB and 2,000 features; WGS84 longitude/latitude.")
            if upload is not None and st.button("Import this GeoJSON", type="primary", use_container_width=True):
                try:
                    features, digest = parse_geojson(upload.getvalue())
                    st.session_state.plan_features = features
                    st.session_state.plan_sha = digest
                    st.session_state.plan_source_type = "GeoJSON"
                    st.session_state.selected_feature = features[0]["id"]
                    st.session_state.pop("feature_selector", None)
                    st.rerun()
                except ValueError as err:
                    st.error(str(err))
        with right:
            st.markdown("**GeoJSON import**")
            st.caption("Upload a WGS84 FeatureCollection of individual plan elements. "
                       "Existing source properties are preserved for inspection but never interpreted as classifications.")
            st.caption("Each feature begins unclassified, even if its source layer has a descriptive name.")
    else:
        st.info("DXF geometry is imported without semantic interpretation. Confirm site, drawing units, "
                "coordinate placement and alignment before creating the classification inventory.")
        with left:
            st.markdown("**Step 1 · Define project site**")
            st.caption("Use a separately verified project-boundary polygon. DXF drawings often use local CAD coordinates.")
        site = site_boundary_ui()
        current_site_key = json.dumps(site, sort_keys=True) if site is not None else None
        if st.session_state.get("v3_import_site_key") != current_site_key:
            if st.session_state.get("plan_source_type") == "DXF":
                st.session_state.plan_features = []
                st.session_state.plan_sha = None
                st.session_state.selected_feature = None
                st.session_state.pop("feature_selector", None)
            st.session_state["v3_import_site_key"] = current_site_key
        if site is not None:
            st.markdown("**Step 2 · Place DXF master plan**")
            placement = dxf_placement_ui(site, key="v3_dxf")
            if placement is not None:
                st.success(f"Placement accepted · {len(placement.feature_collection['features']):,} CAD-derived geometries")
                st.caption("CAD layers and blocks remain source metadata. The original file is not altered. "
                           "Only currently selected visible layers will be imported.")
                include_boundary = st.checkbox("Include the CAD site-boundary layer as a classifiable feature",
                                               value=False, key="v3_include_site_outline")
                chosen_layers = set(st.session_state.get("v3_dxf_layers", []))
                boundary_layer = st.session_state.get("v3_dxf_boundary_layer")
                selected_features = [f for f in placement.feature_collection["features"]
                                     if f["properties"].get("layer") in chosen_layers and
                                     (include_boundary or f["properties"].get("layer") != boundary_layer)]
                st.caption(f"{len(selected_features):,} features from selected CAD layers will enter manual classification. "
                           "Unselected layers and the optional site-outline layer are excluded.")
                if st.button("Import accepted DXF for manual classification", type="primary", key="import_placed_dxf",
                             disabled=not selected_features):
                    try:
                        raw = json.dumps({"type": "FeatureCollection", "features": selected_features}).encode("utf-8")
                        features, digest = parse_geojson(raw)
                        st.session_state.plan_features = features
                        st.session_state.plan_sha = digest
                        st.session_state.plan_source_type = "DXF"
                        st.session_state.selected_feature = features[0]["id"]
                        st.session_state.pop("feature_selector", None)
                        st.rerun()
                    except ValueError as err:
                        st.error("DXF geometry not accepted: " + str(err))
                if len(placement.feature_collection["features"]) > 2000:
                    st.warning("This DXF has over 2,000 features. Export only relevant layers or simplify it "
                               "in CAD before using the classification prototype.")
    if st.session_state.plan_features:
        st.success(f"Master plan imported from {st.session_state.get('plan_source_type', 'file')} · "
                   f"{len(st.session_state.plan_features):,} features")
        st.caption("All features require explicit classification. Re-importing replaces session assignments; "
                   "export the classified master plan first.")
    else:
        st.info("Import a master plan to activate individual feature classification.")

with classify_tab:
    st.subheader("You classify. ClimaForma measures.")
    if not st.session_state.plan_features:
        st.info("First upload and import a GeoJSON master plan in Project & Site.")
    else:
        features = st.session_state.plan_features
        map_col, inspector = st.columns([1.72, 1], gap="large")
        with map_col:
            st.caption("Click a feature to select it. Unclassified features are gray; manually classified features use category colors.")
            shapes = [shape(f["geometry"]) for f in features]
            full_bounds = [min(g.bounds[0] for g in shapes), min(g.bounds[1] for g in shapes),
                           max(g.bounds[2] for g in shapes), max(g.bounds[3] for g in shapes)]
            center = [(full_bounds[1]+full_bounds[3])/2, (full_bounds[0]+full_bounds[2])/2]
            m = folium.Map(location=center, zoom_start=17, tiles="CartoDB positron", control_scale=True)
            for f in features:
                color = PALETTE[f["class"]]
                selected = f["id"] == st.session_state.selected_feature
                geo = {"type":"Feature", "geometry":f["geometry"],
                       "properties":{"cf_selector":f["id"],"cf_label":f["id"] + (" · " + f["class"] if f["class"] else " · Unclassified")}}
                folium.GeoJson(geo, name=f["id"],
                               style_function=lambda _, col=color, sel=selected: {"color":"#0e665d" if sel else col,
                                   "weight":5 if sel else 3,"fillColor":col,"fillOpacity":0.42 if sel else 0.28},
                               marker=folium.CircleMarker(radius=7, color=color, fill=True, fill_opacity=.95),
                               tooltip=folium.GeoJsonTooltip(fields=["cf_selector"], labels=False, sticky=True)
                              ).add_to(m)
            if full_bounds[0] != full_bounds[2] and full_bounds[1] != full_bounds[3]:
                m.fit_bounds([[full_bounds[1], full_bounds[0]],[full_bounds[3], full_bounds[2]]], padding=(20,20))
            elif full_bounds[0] == full_bounds[2] and full_bounds[1] == full_bounds[3]:
                m.location = center
            result = st_folium(m, height=510, use_container_width=True,
                               key="v3_classification_map", returned_objects=["last_object_clicked_tooltip"])
            tooltip = result.get("last_object_clicked_tooltip") if isinstance(result,dict) else None
            if isinstance(tooltip,str):
                match = re.search(r"CF-\d{4}", tooltip)
                if match and match.group(0) != st.session_state.selected_feature:
                    st.session_state.selected_feature = match.group(0)
                    st.session_state.pop("feature_selector",None)
                    st.rerun()
            st.caption("Feature clicking depends on the Folium event bridge; the feature selector on the right always provides a direct alternative.")
        with inspector:
            st.markdown("#### Feature inspector")
            ids = [f["id"] for f in features]
            sel = st.session_state.selected_feature
            if sel not in ids:
                sel = ids[0]
                st.session_state.selected_feature = sel
            if st.session_state.get("feature_selector") not in ids:
                st.session_state["feature_selector"] = sel
            chosen_id = st.selectbox("Select a geometry", ids, key="feature_selector",
                                      format_func=lambda ident: ident + " · " + next(f["geometry_type"] for f in features if f["id"]==ident))
            if chosen_id != st.session_state.selected_feature:
                st.session_state.selected_feature = chosen_id
                st.rerun()
            f = next(f for f in features if f["id"] == chosen_id)
            st.caption(f"Geometry: {f['geometry_type']} | Source feature ID: {f['source_id'] or 'not provided'}")
            if f["properties"].get("layer") is not None:
                st.caption(f"CAD source layer: {f['properties']['layer']} · CAD entity: {f['properties'].get('kind', 'unknown')}")
            if f["area_m2"] is not None:
                st.metric("Geometry area",f"{f['area_m2']:,.1f} m²")
            elif f["length_m"] is not None:
                st.metric("Line length",f"{f['length_m']:,.1f} m")
                st.caption("Line areas cannot be calculated without an explicitly supplied width.")
            else:
                st.metric("Geometry count", "1 feature")
                st.caption("A tree point is not an observed or assumed canopy area.")
            st.caption("Current assignment: " + (f["class"] or "UNCLASSIFIED"))
            with st.form(f"manual_feature_{chosen_id}"):
                selected_class = st.selectbox("Assign this feature manually", ["— Select a class —"] + list(CLASSES),
                                              index=(CLASSES.index(f["class"])+1 if f["class"] else 0))
                note = st.text_area("Optional design notes / evidence", value=f["class_note"],height=90,max_chars=500)
                submit = st.form_submit_button("Save manual classification",type="primary",use_container_width=True)
            if submit:
                if selected_class not in CLASSES:
                    st.warning("Select a class before saving.")
                else:
                    st.session_state.plan_features = assign_class(features, chosen_id,selected_class,note)
                    st.success("Saved explicit manual assignment.")
                    st.rerun()
            if f["class"]:
                if st.button("Reset feature to unclassified",key=f"reset_{chosen_id}",use_container_width=True):
                    st.session_state.plan_features = clear_class(features,chosen_id)
                    st.rerun()
        st.progress(summary["completion_pct"]/100, text=f"{summary['classified']} of {summary['features']} features manually classified")
        st.download_button("Export manually classified GeoJSON", export_geojson(st.session_state.plan_features),
                           file_name="climaforma_classified_masterplan.geojson", mime="application/geo+json",
                           use_container_width=False)

with baseline_tab:
    st.subheader("Baseline master-plan composition")
    if not st.session_state.plan_features:
        st.info("Import and manually classify a master plan first.")
    else:
        st.markdown("The following are **design-geometry quantities**, not Earth-observed environmental performance.")
        if summary["unclassified"]:
            st.warning(f"{summary['unclassified']} features remain unclassified. Totals are incomplete.")
        rows=[]
        for cls in CLASSES:
            rows.append({"Manual class":cls, "Feature count":summary["counts"].get(cls,0),
                         "Polygon area (m²)":summary["areas_m2"].get(cls,0.0),
                         "Features without area":summary["non_area_features"].get(cls,0)})
        st.dataframe(pd.DataFrame(rows),use_container_width=True,hide_index=True)
        st.caption("Area totals are sums by class, not mutually exclusive site coverages. Overlapping features and missing site boundaries require validation before site percentages can be reported.")
        st.markdown("#### Environmental context and microclimate")
        st.info("The EO and modelling engines are intentionally not connected in this UI preview. Future phases will restore observed Landsat LST and Sentinel-2 NDVI/NDBI as site context, then assess plan performance using a documented, validated method. No invented UHI or future NDVI values are displayed.")

with options_tab:
    st.subheader("Design options")
    st.markdown('<div class="cf-panel"><div class="cf-label">Next implementation stage</div><div class="cf-value">Edit a copy or upload a revised plan</div><div class="cf-note">The original, manually classified master plan will be immutable. Each alternative will have independent geometries, classes and version history.</div></div>',unsafe_allow_html=True)
    st.markdown("**Pathway A:** Duplicate the baseline and edit, add, move, remove or reclassify design features.")
    st.markdown("**Pathway B:** Upload another GeoJSON and classify its features independently.")
    st.caption("These controls are deliberately not active in this UI-first release; no simulated changes are implied.")

with compare_tab:
    st.subheader("Compare design performance")
    st.markdown("The comparison stage will show two different outputs, clearly separated:")
    st.markdown("**Physical changes:** building footprints, vegetation and grass coverage, roads, paving, water and tree/canopy properties.")
    st.markdown("**Environmental response:** supported expected directional responses and, after model integration and validation, simulated surface temperature, solar exposure, thermal comfort or UHI metrics.")
    st.info("No alternative plans or microclimate simulations exist in the UI prototype yet. This panel will not fabricate values.")

st.divider()
st.caption("ClimaForma v3.0 DXF + GeoJSON prototype • Manual classification only • No automated land-cover labels, future NDVI, UHI, LST, or cooling predictions")
