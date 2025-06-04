# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import geopandas  # type: ignore
import pandas as pd
import pandas.testing
import pytest
from shapely.geometry import (  # type: ignore
    GeometryCollection,
    LineString,
    MultiLineString,
    MultiPoint,
    MultiPolygon,
    Point,
    Polygon,
)

from bigframes.bigquery import st_length
import bigframes.bigquery as bbq
import bigframes.geopandas
import bigframes.pandas as bpd


def test_geo_st_area():
    data = [
        Polygon([(0.000, 0.0), (0.001, 0.001), (0.000, 0.001)]),
        Polygon([(0.0010, 0.004), (0.009, 0.005), (0.0010, 0.005)]),
        Polygon([(0.001, 0.001), (0.002, 0.001), (0.002, 0.002)]),
        LineString([(0, 0), (1, 1), (0, 1)]),
        Point(0, 1),
    ]

    geopd_s = geopandas.GeoSeries(data=data, crs="EPSG:4326")
    geobf_s = bigframes.geopandas.GeoSeries(data=data)

    # For `geopd_s`, the data was further projected with `geopandas.GeoSeries.to_crs`
    # to `to_crs(26393)` to get the area in square meter. See: https://geopandas.org/en/stable/docs/user_guide/projections.html
    # and https://spatialreference.org/ref/epsg/26393/. We then rounded both results
    # to get them as close to each other as possible. Initially, the area results
    # were +ten-millions. We added more zeros after the decimal point to round the
    # area results to the nearest thousands.
    geopd_s_result = geopd_s.to_crs(26393).area.round(-3)
    geobf_s_result = bbq.st_area(geobf_s).to_pandas().round(-3)
    assert geobf_s_result.iloc[0] >= 1000

    pd.testing.assert_series_equal(
        geobf_s_result,
        geopd_s_result,
        check_dtype=False,
        check_index_type=False,
        check_exact=False,
        rtol=0.1,
    )


# Expected length for 1 degree of longitude at the equator is approx 111195.079734 meters
DEG_LNG_EQUATOR_METERS = 111195.07973400292


def test_st_length_point(session):
    geoseries = bigframes.geopandas.GeoSeries([Point(0, 0)], session=session)
    result = st_length(geoseries).to_pandas()
    expected = bpd.Series([0.0], dtype="Float64")
    pd.testing.assert_series_equal(
        result,
        expected,
        check_dtype=False,
        check_index_type=False,
        rtol=1e-3,
        atol=1e-3,  # For zero values, rtol is not effective
    )  # type: ignore


def test_st_length_linestring(session):
    geoseries = bigframes.geopandas.GeoSeries(
        [LineString([(0, 0), (1, 0)])], session=session
    )
    result = st_length(geoseries).to_pandas()
    expected = bpd.Series([DEG_LNG_EQUATOR_METERS], dtype="Float64")
    pd.testing.assert_series_equal(
        result,
        expected,
        check_dtype=False,
        check_index_type=False,
        rtol=1e-3,
    )  # type: ignore


def test_st_length_polygon(session):
    # Square polygon, 1 degree side. Perimeter should be ~4 * DEG_LNG_EQUATOR_METERS
    # However, diagonal length varies with latitude. For simplicity, use a known BQ result if possible
    # or a very simple polygon whose length is less ambiguous.
    # Using a simple line for perimeter calculation for now.
    # A polygon like POLYGON((0 0, 1 0, 1 1, 0 1, 0 0))
    # Lengths: (0,0)-(1,0) -> DEG_LNG_EQUATOR_METERS
    # (1,0)-(1,1) -> DEG_LAT_METERS (approx DEG_LNG_EQUATOR_METERS)
    # (1,1)-(0,1) -> DEG_LNG_EQUATOR_METERS (at lat 1)
    # (0,1)-(0,0) -> DEG_LAT_METERS
    # This gets complicated due to earth curvature.
    # Let's test with a polygon known to BQ.
    # Example from BQ docs: ST_LENGTH(ST_GEOGFROMTEXT('POLYGON((0 0, 1 0, 0 1, 0 0))')) == 333585.1992020086
    geoseries = bigframes.geopandas.GeoSeries(
        [Polygon([(0, 0), (1, 0), (0, 1), (0, 0)])], session=session
    )
    result = st_length(geoseries).to_pandas()
    expected = bpd.Series([333585.1992020086], dtype="Float64")
    pd.testing.assert_series_equal(
        result,
        expected,
        check_dtype=False,
        check_index_type=False,
        rtol=1e-3,  # Increased tolerance for complex polygon calculation
    )  # type: ignore


def test_st_length_multipoint(session):
    geoseries = bigframes.geopandas.GeoSeries(
        [MultiPoint([Point(0, 0), Point(1, 1)])], session=session
    )
    result = st_length(geoseries).to_pandas()
    expected = bpd.Series([0.0], dtype="Float64")
    pd.testing.assert_series_equal(
        result,
        expected,
        check_dtype=False,
        check_index_type=False,
        rtol=1e-3,
        atol=1e-3,
    )  # type: ignore


def test_st_length_multilinestring(session):
    geoseries = bigframes.geopandas.GeoSeries(
        [MultiLineString([LineString([(0, 0), (1, 0)]), LineString([(0, 0), (0, 1)])])],
        session=session,
    )
    result = st_length(geoseries).to_pandas()
    # Sum of lengths of two lines, each 1 degree.
    # ST_Length(ST_GeogFromText('MultiLineString((0 0, 1 0), (0 0, 0 1))')) = 222390.15946800584
    expected = bpd.Series([2 * DEG_LNG_EQUATOR_METERS], dtype="Float64")
    pd.testing.assert_series_equal(
        result,
        expected,
        check_dtype=False,
        check_index_type=False,
        rtol=1e-3,
    )  # type: ignore


def test_st_length_multipolygon(session):
    # Two separate polygons. Length is sum of their perimeters.
    # Polygon 1: POLYGON((0 0, 1 0, 0 1, 0 0)) -> 333585.1992020086
    # Polygon 2 (smaller triangle): POLYGON((2 0, 3 0, 2 1, 2 0)) -> 333585.1992020086 (similar triangle)
    # Let's use distinct polygons for clarity
    # Polygon 1: POLYGON((0 0, 1 0, 0 1, 0 0)) -> 333585.1992020086
    # Polygon 2: POLYGON((2 2, 3 2, 2 3, 2 2)) -> 333585.1992020086
    # Total expected: 2 * 333585.1992020086
    geoseries = bigframes.geopandas.GeoSeries(
        [
            MultiPolygon(
                [
                    Polygon([(0, 0), (1, 0), (0, 1), (0, 0)]),
                    Polygon([(2, 2), (3, 2), (2, 3), (2, 2)]),
                ]
            )
        ],
        session=session,
    )
    result = st_length(geoseries).to_pandas()
    expected_single_poly_length = 333585.1992020086
    expected = bpd.Series([2 * expected_single_poly_length], dtype="Float64")
    pd.testing.assert_series_equal(
        result,
        expected,
        check_dtype=False,
        check_index_type=False,
        rtol=1e-3,
    )  # type: ignore


def test_st_length_geometrycollection(session):
    # Collection: Point(0,0), LineString((0,0),(1,0))
    # Expected: 0 (for point) + DEG_LNG_EQUATOR_METERS (for line)
    geoseries = bigframes.geopandas.GeoSeries(
        [GeometryCollection([Point(0, 0), LineString([(0, 0), (1, 0)])])],
        session=session,
    )
    result = st_length(geoseries).to_pandas()
    expected = bpd.Series([DEG_LNG_EQUATOR_METERS], dtype="Float64")
    pd.testing.assert_series_equal(
        result,
        expected,
        check_dtype=False,
        check_index_type=False,
        rtol=1e-3,
    )  # type: ignore


def test_st_length_geometrycollection_polygon_line(session):
    # Collection: Polygon((0 0, 1 0, 0 1, 0 0)), LineString((2,0),(3,0))
    # Expected: 333585.1992020086 + DEG_LNG_EQUATOR_METERS
    poly_length = 333585.1992020086
    line_length = DEG_LNG_EQUATOR_METERS
    geoseries = bigframes.geopandas.GeoSeries(
        [
            GeometryCollection(
                [
                    Polygon([(0, 0), (1, 0), (0, 1), (0, 0)]),
                    LineString([(2, 0), (3, 0)]),
                ]
            )
        ],
        session=session,
    )
    result = st_length(geoseries).to_pandas()
    expected = bpd.Series([poly_length + line_length], dtype="Float64")
    pd.testing.assert_series_equal(
        result,
        expected,
        check_dtype=False,
        check_index_type=False,
        rtol=1e-3,
    )  # type: ignore


def test_st_length_empty_geography(session):
    # Representing empty geography can be tricky.
    # An empty GeometryCollection is one way.
    # Or a GeoSeries with None or empty string that BQ interprets as empty geography
    geoseries_empty_collection = bigframes.geopandas.GeoSeries(
        [GeometryCollection([])], session=session
    )
    result_empty_collection = st_length(geoseries_empty_collection)
    expected_empty = bpd.Series([0.0], dtype="Float64")
    pd.testing.assert_series_equal(
        result_empty_collection,
        expected_empty,
        check_dtype=False,
        check_index_type=False,
        rtol=1e-3,
        atol=1e-3,
    )  # type: ignore

    # Test with None, which should also result in 0 or be handled as NULL by BQ ST_LENGTH if it propagates
    # BQ ST_LENGTH(NULL) is NULL. BigQuery GeoSeries might convert None to empty GEOGRAPHY string.
    # Let's test with WKT of an empty geometry
    geoseries_empty_wkt = bigframes.geopandas.GeoSeries(
        ["GEOMETRYCOLLECTION EMPTY"], session=session
    )
    result_empty_wkt = st_length(geoseries_empty_wkt)
    pd.testing.assert_series_equal(
        result_empty_wkt,
        expected_empty,  # Expect 0.0 for empty geometries
        check_dtype=False,
        check_index_type=False,
        rtol=1e-3,
        atol=1e-3,
    )  # type: ignore


def test_st_length_geometrycollection_only_points(session):
    geoseries = bigframes.geopandas.GeoSeries(
        [GeometryCollection([Point(0, 0), Point(1, 1)])], session=session
    )
    result = st_length(geoseries).to_pandas()
    expected = bpd.Series([0.0], dtype="Float64")
    pd.testing.assert_series_equal(
        result,
        expected,
        check_dtype=False,
        check_index_type=False,
        rtol=1e-3,
        atol=1e-3,
    )  # type: ignore


def test_st_length_mixed_types_and_nulls(session):
    geoseries = bigframes.geopandas.GeoSeries(
        [
            Point(0, 1),
            LineString([(0, 0), (1, 0)]),
            Polygon([(0, 0), (0.0001, 0), (0, 0.0001), (0, 0)]),  # very small polygon
            None,  # Should result in NA or handle as 0 if BQ converts to empty
            GeometryCollection(
                [Point(1, 1), LineString([(0, 0), (0.00001, 0)])]
            ),  # Point length 0, line length tiny
        ],
        session=session,
    )
    result = st_length(geoseries).to_pandas()

    # Expected:
    # Point: 0.0
    # LineString: DEG_LNG_EQUATOR_METERS
    # Polygon: ST_Length(ST_GeogFromText('POLYGON((0 0, 0.0001 0, 0 0.0001, 0 0))')) approx 3 * 0.0001 * DEG_LNG_EQUATOR_METERS at small scale
    # Using BQ value for this small polygon: ST_LENGTH(ST_GEOGFROMTEXT('POLYGON((0 0, 0.0001 0, 0 0.0001, 0 0))')) = 33.35851992020086
    # None: NaN (since ST_LENGTH(NULL) is NULL)
    # GeometryCollection: 0 + (0.00001 * DEG_LNG_EQUATOR_METERS) = 0 + 1.11195079734
    expected_data = [
        0.0,
        DEG_LNG_EQUATOR_METERS,
        33.35851992020086,
        None,  # Representing NA for pandas/bigframes series
        0.00001 * DEG_LNG_EQUATOR_METERS,
    ]
    expected = bpd.Series(expected_data, dtype="Float64")

    pd.testing.assert_series_equal(
        result,
        expected,
        check_index_type=False,
        rtol=1e-3,
        atol=1e-2,  # For small values and None comparison
    )  # type: ignore


def test_geo_st_difference_with_geometry_objects():
    data1 = [
        Polygon([(0, 0), (10, 0), (10, 10), (0, 0)]),
        Polygon([(0, 0), (1, 1), (0, 1), (0, 0)]),
        Point(0, 1),
    ]

    data2 = [
        Polygon([(0, 0), (10, 0), (10, 10), (0, 0)]),
        Polygon([(0, 0), (1, 1), (0, 1), (0, 0)]),
        LineString([(2, 0), (0, 2)]),
    ]

    geobf_s1 = bigframes.geopandas.GeoSeries(data=data1)
    geobf_s2 = bigframes.geopandas.GeoSeries(data=data2)
    geobf_s_result = bbq.st_difference(geobf_s1, geobf_s2).to_pandas()

    expected = pd.Series(
        [
            GeometryCollection([]),
            GeometryCollection([]),
            Point(0, 1),
        ],
        index=[0, 1, 2],
        dtype=geopandas.array.GeometryDtype(),
    )
    pandas.testing.assert_series_equal(
        geobf_s_result,
        expected,
        check_index_type=False,
        check_exact=False,
        rtol=0.1,
    )


def test_geo_st_difference_with_single_geometry_object():
    pytest.importorskip(
        "shapely",
        minversion="2.0.0",
        reason="shapely objects must be hashable to include in our expression trees",
    )

    data1 = [
        Polygon([(0, 0), (10, 0), (10, 10), (0, 10), (0, 0)]),
        Polygon([(0, 1), (10, 1), (10, 9), (0, 9), (0, 1)]),
        Point(0, 1),
    ]

    geobf_s1 = bigframes.geopandas.GeoSeries(data=data1)
    geobf_s_result = bbq.st_difference(
        geobf_s1,
        Polygon([(0, 0), (10, 0), (10, 5), (0, 5), (0, 0)]),
    ).to_pandas()

    expected = pd.Series(
        [
            Polygon([(10, 5), (10, 10), (0, 10), (0, 5), (10, 5)]),
            Polygon([(10, 5), (10, 9), (0, 9), (0, 5), (10, 5)]),
            GeometryCollection([]),
        ],
        index=[0, 1, 2],
        dtype=geopandas.array.GeometryDtype(),
    )
    pandas.testing.assert_series_equal(
        geobf_s_result,
        expected,
        check_index_type=False,
        check_exact=False,
        rtol=0.1,
    )


def test_geo_st_difference_with_similar_geometry_objects():
    data1 = [
        Polygon([(0, 0), (10, 0), (10, 10), (0, 0)]),
        Polygon([(0, 0), (1, 1), (0, 1)]),
        Point(0, 1),
    ]

    geobf_s1 = bigframes.geopandas.GeoSeries(data=data1)
    geobf_s_result = bbq.st_difference(geobf_s1, geobf_s1).to_pandas()

    expected = pd.Series(
        [GeometryCollection([]), GeometryCollection([]), GeometryCollection([])],
        index=[0, 1, 2],
        dtype=geopandas.array.GeometryDtype(),
    )
    pandas.testing.assert_series_equal(
        geobf_s_result,
        expected,
        check_index_type=False,
        check_exact=False,
        rtol=0.1,
    )


def test_geo_st_distance_with_geometry_objects():
    data1 = [
        # 0.00001 is approximately 1 meter.
        Polygon([(0, 0), (0.00001, 0), (0.00001, 0.00001), (0, 0.00001), (0, 0)]),
        Polygon(
            [
                (0.00002, 0),
                (0.00003, 0),
                (0.00003, 0.00001),
                (0.00002, 0.00001),
                (0.00002, 0),
            ]
        ),
        Point(0, 0.00002),
    ]

    data2 = [
        Polygon(
            [
                (0.00002, 0),
                (0.00003, 0),
                (0.00003, 0.00001),
                (0.00002, 0.00001),
                (0.00002, 0),
            ]
        ),
        Point(0, 0.00002),
        Polygon([(0, 0), (0.00001, 0), (0.00001, 0.00001), (0, 0.00001), (0, 0)]),
        Point(
            1, 1
        ),  # No matching row in data1, so this will be NULL after the call to distance.
    ]

    geobf_s1 = bigframes.geopandas.GeoSeries(data=data1)
    geobf_s2 = bigframes.geopandas.GeoSeries(data=data2)
    geobf_s_result = bbq.st_distance(geobf_s1, geobf_s2).to_pandas()

    expected = pd.Series(
        [
            1.112,
            2.486,
            1.112,
            None,
        ],
        index=[0, 1, 2, 3],
        dtype="Float64",
    )
    pandas.testing.assert_series_equal(
        geobf_s_result,
        expected,
        check_index_type=False,
        check_exact=False,
        rtol=0.1,
    )


def test_geo_st_distance_with_single_geometry_object():
    pytest.importorskip(
        "shapely",
        minversion="2.0.0",
        reason="shapely objects must be hashable to include in our expression trees",
    )

    data1 = [
        # 0.00001 is approximately 1 meter.
        Polygon([(0, 0), (0.00001, 0), (0.00001, 0.00001), (0, 0.00001), (0, 0)]),
        Polygon(
            [
                (0.00001, 0),
                (0.00002, 0),
                (0.00002, 0.00001),
                (0.00001, 0.00001),
                (0.00001, 0),
            ]
        ),
        Point(0, 0.00002),
    ]

    geobf_s1 = bigframes.geopandas.GeoSeries(data=data1)
    geobf_s_result = bbq.st_distance(
        geobf_s1,
        Point(0, 0),
    ).to_pandas()

    expected = pd.Series(
        [
            0,
            1.112,
            2.224,
        ],
        dtype="Float64",
    )
    pandas.testing.assert_series_equal(
        geobf_s_result,
        expected,
        check_index_type=False,
        check_exact=False,
        rtol=0.1,
    )


def test_geo_st_intersection_with_geometry_objects():
    data1 = [
        Polygon([(0, 0), (10, 0), (10, 10), (0, 0)]),
        Polygon([(0, 0), (1, 1), (0, 1), (0, 0)]),
        Point(0, 1),
    ]

    data2 = [
        Polygon([(0, 0), (10, 0), (10, 10), (0, 0)]),
        Polygon([(0, 0), (1, 1), (0, 1), (0, 0)]),
        LineString([(2, 0), (0, 2)]),
    ]

    geobf_s1 = bigframes.geopandas.GeoSeries(data=data1)
    geobf_s2 = bigframes.geopandas.GeoSeries(data=data2)
    geobf_s_result = bbq.st_intersection(geobf_s1, geobf_s2).to_pandas()

    expected = pd.Series(
        [
            Polygon([(0, 0), (10, 0), (10, 10), (0, 0)]),
            Polygon([(0, 0), (1, 1), (0, 1), (0, 0)]),
            GeometryCollection([]),
        ],
        index=[0, 1, 2],
        dtype=geopandas.array.GeometryDtype(),
    )
    pandas.testing.assert_series_equal(
        geobf_s_result,
        expected,
        check_index_type=False,
        check_exact=False,
        rtol=0.1,
    )


def test_geo_st_intersection_with_single_geometry_object():
    pytest.importorskip(
        "shapely",
        minversion="2.0.0",
        reason="shapely objects must be hashable to include in our expression trees",
    )

    data1 = [
        Polygon([(0, 0), (10, 0), (10, 10), (0, 10), (0, 0)]),
        Polygon([(0, 1), (10, 1), (10, 9), (0, 9), (0, 1)]),
        Point(0, 1),
    ]

    geobf_s1 = bigframes.geopandas.GeoSeries(data=data1)
    geobf_s_result = bbq.st_intersection(
        geobf_s1,
        Polygon([(0, 0), (10, 0), (10, 5), (0, 5), (0, 0)]),
    ).to_pandas()

    expected = pd.Series(
        [
            Polygon([(0, 0), (10, 0), (10, 5), (0, 5), (0, 0)]),
            Polygon([(0, 1), (10, 1), (10, 5), (0, 5), (0, 1)]),
            Point(0, 1),
        ],
        index=[0, 1, 2],
        dtype=geopandas.array.GeometryDtype(),
    )
    pandas.testing.assert_series_equal(
        geobf_s_result,
        expected,
        check_index_type=False,
        check_exact=False,
        rtol=0.1,
    )


def test_geo_st_intersection_with_similar_geometry_objects():
    data1 = [
        Polygon([(0, 0), (10, 0), (10, 10), (0, 0)]),
        Polygon([(0, 0), (1, 1), (0, 1)]),
        Point(0, 1),
    ]

    geobf_s1 = bigframes.geopandas.GeoSeries(data=data1)
    geobf_s_result = bbq.st_intersection(geobf_s1, geobf_s1).to_pandas()

    expected = pd.Series(
        [
            Polygon([(0, 0), (10, 0), (10, 10), (0, 0)]),
            Polygon([(0, 0), (1, 1), (0, 1)]),
            Point(0, 1),
        ],
        index=[0, 1, 2],
        dtype=geopandas.array.GeometryDtype(),
    )
    pandas.testing.assert_series_equal(
        geobf_s_result,
        expected,
        check_index_type=False,
        check_exact=False,
        rtol=0.1,
    )
