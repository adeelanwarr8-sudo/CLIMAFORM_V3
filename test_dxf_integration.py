"""Pure DXF geometry/placement tests without requiring Streamlit server."""
import io
import json
import sys
import types
import importlib.util
from pathlib import Path

import ezdxf
from shapely.geometry import Polygon

ROOT = Path(__file__).resolve().parents[1]
# UI imports are not needed for geometry functions in this test runner.
if 'streamlit' not in sys.modules:
    try:
        import streamlit
    except ModuleNotFoundError:
        sys.modules['streamlit'] = types.ModuleType('streamlit')
if 'streamlit_folium' not in sys.modules:
    try:
        import streamlit_folium
    except ModuleNotFoundError:
        mock = types.ModuleType('streamlit_folium')
        mock.st_folium = lambda *a,**kw: None
        sys.modules['streamlit_folium'] = mock
spec = importlib.util.spec_from_file_location('climaforma_dxf', ROOT/'climaforma_dxf.py')
mod = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = mod
spec.loader.exec_module(mod)

from masterplan import parse_geojson, inventory, assign_class


def sample_dxf():
    doc = ezdxf.new('R2010')
    doc.header['$INSUNITS'] = 6
    m = doc.modelspace()
    doc.blocks.new('TREE_SYMBOL').add_circle((0,0),2)
    m.add_lwpolyline([(0,0),(30,0),(30,20),(0,20)], close=True, dxfattribs={'layer':'SITE'})
    m.add_lwpolyline([(5,5),(14,5),(14,12),(5,12)], close=True, dxfattribs={'layer':'BUILDING'})
    m.add_blockref('TREE_SYMBOL',(18,12),dxfattribs={'layer':'SYMBOLS'})
    txt = io.StringIO(); doc.write(txt)
    return txt.getvalue().encode('utf-8')


def site():
    return {'type':'Polygon','coordinates':[[[2.0,41.0],[2.00036,41.0],[2.00036,41.00018],[2.0,41.00018],[2.0,41.0]]]}


def test_read_dxf_preserves_layers_and_block_is_single_feature():
    plan=mod.read_dxf(sample_dxf())
    assert len(plan.features)==3
    assert plan.unit_scale==1
    assert {f['layer'] for f in plan.features}=={'SITE','BUILDING','SYMBOLS'}
    assert len([f for f in plan.features if f['kind']=='block-insert'])==1


def test_aligned_dxf_to_wgs84_remains_unclassified():
    plan=mod.read_dxf(sample_dxf())
    al,_ = mod.initial_alignment(plan,site(),'SITE')
    placed=mod.place(plan,al,site(),'SITE')
    normalized,hash_=parse_geojson(json.dumps(placed.feature_collection).encode())
    assert len(normalized)==3
    assert all(f['class'] is None for f in normalized)
    assert inventory(normalized)['unclassified']==3
    assert normalized[1]['properties']['layer']=='BUILDING'
    changed=assign_class(normalized,normalized[1]['id'],'Building / roof')
    assert changed[1]['class']=='Building / roof'
    assert normalized[1]['class'] is None


def test_projected_crs_no_double_scaling():
    plan=mod.read_dxf(sample_dxf())
    # Valid projected-origin coordinates near Barcelona in EPSG:25831, including a non-1 INSUNITS.
    from shapely import affinity
    plan.features=[{**f,'geom':affinity.translate(f['geom'], 432000, 4582000)} for f in plan.features]
    plan.unit_scale=0.001
    placed=mod.place_by_crs(plan,25831)
    coords=placed.feature_collection['features'][2]['geometry']['coordinates']
    assert 1 < coords[0] < 4
    assert 40 < coords[1] < 43
    assert any('INSUNITS' in n for n in placed.notes)


def test_geometric_extent_and_accuracy():
    plan=mod.read_dxf(sample_dxf())
    al,_=mod.initial_alignment(plan,site(),'SITE')
    placed=mod.place(plan,al,site(),'SITE')
    features,_=parse_geojson(json.dumps(placed.feature_collection).encode())
    assert features[0]['area_m2'] > 500
    assert features[1]['area_m2'] > 50
