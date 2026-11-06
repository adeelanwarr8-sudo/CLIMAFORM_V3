import json
import pytest
from masterplan import parse_geojson, assign_class, clear_class, inventory, export_geojson


def make_raw(features):
    return json.dumps({"type":"FeatureCollection","features":features}).encode()


def feature(geom, props=None):
    return {"type":"Feature","geometry":geom,"properties":props or {}}


def test_manual_only():
    fs, _ = parse_geojson(make_raw([feature({"type":"Point","coordinates":[2.2,41.4]}, {"class":"Tree"})]))
    assert fs[0]["class"] is None  # Source metadata must NEVER classify for user.
    assert inventory(fs)["unclassified"] == 1


def test_manual_assignment_and_export():
    fs,_ = parse_geojson(make_raw([feature({"type":"Point","coordinates":[2.2,41.4]})]))
    edited=assign_class(fs,"CF-0001","Tree / canopy","user supplied")
    assert fs[0]["class"] is None
    assert edited[0]["class"] == "Tree / canopy"
    data=json.loads(export_geojson(edited))
    assert data["features"][0]["properties"]["cf_manual_class"] == "Tree / canopy"
    reset=clear_class(edited,"CF-0001")
    assert reset[0]["class"] is None


def test_invalid_class_rejected():
    fs,_=parse_geojson(make_raw([feature({"type":"Point","coordinates":[2.2,41.4]})]))
    with pytest.raises(ValueError): assign_class(fs,"CF-0001","Automatic NDVI Tree")


def test_polygon_area():
    poly={"type":"Polygon","coordinates":[[[2.2,41.4],[2.201,41.4],[2.201,41.401],[2.2,41.401],[2.2,41.4]]]}
    fs,_=parse_geojson(make_raw([feature(poly)]))
    assert fs[0]["area_m2"] > 0


def test_reject_invalid():
    with pytest.raises(ValueError): parse_geojson(b"bad")
    with pytest.raises(ValueError): parse_geojson(make_raw([feature({"type":"Point","coordinates":[190,41.4]})]))


def test_line_has_length_not_area():
    fs,_ = parse_geojson(make_raw([feature({"type":"LineString","coordinates":[[2.2,41.4],[2.201,41.401]]})]))
    assert fs[0]["length_m"] > 0 and fs[0]["area_m2"] is None


def test_source_attributes_preserved():
    fs,_ = parse_geojson(make_raw([feature({"type":"Point","coordinates":[2.2,41.4]}, {"name":"Tree 1"})]))
    assert fs[0]["properties"]["name"] == "Tree 1"


def test_3d_points_normalized():
    fs,_ = parse_geojson(make_raw([feature({"type":"Point","coordinates":[2.2,41.4,100]})]))
    assert len(fs[0]["geometry"]["coordinates"]) == 2
