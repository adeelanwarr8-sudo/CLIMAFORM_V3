# ClimaForma v3.0 — UI-first proof of workflow

This is a **new standalone Streamlit app**. It must not replace the verified
ClimaForma v2 deployment until its user interface is accepted.

## Start locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

Deploy in a **separate GitHub repository or separate Streamlit Cloud app**.

## What works

- Five-stage master-plan-first navigation and professional sidebar.
- Import WGS84 / CRS84 GeoJSON FeatureCollection (up to 5 MB, 2,000 features).
- Render features and select them by map click or inspector dropdown.
- **No auto classification:** source feature attributes are kept, never used as classes.
- Explicit feature-by-feature assignment of building, tree, grass, road, paved walk,
  other hardscape, water, bare ground, or other.
- Live manual-classification progress, polygon area and line length.
- Export classified GeoJSON with original geometry, user-assigned class, notes.
- Geometric baseline inventory, with explicit warning for incomplete classification
  and missing site boundary / overlapping features.

## Not included yet

- Separate verified project boundary upload and overlap accounting.
- Plan geometry editing, revised master-plan upload, scenario comparison.
- EO retrieval and thermal/NDVI/UHI predictive modelling.

These intentionally remain future, non-functional stages. No fake simulated
indicators are displayed. Existing ClimaForma v2 analytical code is preserved
outside this new UI proof of workflow.

## Browser acceptance test

1. Import a GeoJSON containing multiple geometries. Verify all start unclassified.
2. Click a feature in the map or select its CF ID in the inspector.
3. Assign a building, tree, paved-area, green-space, or water class and save.
4. Verify feature color and classification progress update.
5. Inspect baseline quantities and download GeoJSON; verify cf_manual_class exists.
6. Re-import and verify classification resets. Check all other tabs and confirm no
   unimplemented environmental results are portrayed as actual observations.
