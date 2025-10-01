# Copyright 2024 Google LLC
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

from __future__ import annotations

import geopandas as gpd  # type: ignore
import pandas as pd
import pytest

import bigframes.geopandas as bpd
import geopandas as gpd
import geopandas.testing
import pandas as pd
import pytest


def test_geoseries_is_empty(polars_session):
    session = polars_session
    geometries = [
        "POINT (0 0)",
        "POLYGON EMPTY",
    ]
    gseries = gpd.GeoSeries.from_wkt(geometries)

    bf_gseries = bpd.GeoSeries(gseries, session=session)

    result = bf_gseries.is_empty.to_pandas()
    expected = pd.Series([False, True], dtype="boolean", name="is_empty")

    pd.testing.assert_series_equal(expected, result, check_index=False)


def test_geoseries_union(polars_session):
    session = polars_session
    gseries1 = gpd.GeoSeries.from_wkt(
        [
            "POINT (0 0)",
            "POLYGON ((0 0, 1 0, 1 1, 0 1, 0 0))",
        ]
    )
    gseries2 = gpd.GeoSeries.from_wkt(
        [
            "POINT (1 1)",
            "POLYGON ((2 0, 3 0, 3 1, 2 1, 2 0))",
        ]
    )
    expected_union = gpd.GeoSeries.from_wkt(
        [
            "MULTIPOINT (0 0, 1 1)",
            "MULTIPOLYGON (((0 0, 1 0, 1 1, 0 1, 0 0)), ((2 0, 3 0, 3 1, 2 1, 2 0)))",
        ]
    )

    bf_gseries1 = bpd.GeoSeries(gseries1, session=session)
    bf_gseries2 = bpd.GeoSeries(gseries2, session=session)

    result = bf_gseries1.union(bf_gseries2).to_pandas()
    expected = pd.Series(expected_union, dtype=gpd.array.GeometryDtype())

    gpd.testing.assert_geoseries_equal(result, expected, check_series_type=False)


def test_geoseries_is_valid(polars_session):
    session = polars_session
    geometries = [
        "POLYGON ((0 0, 1 1, 0 1, 0 0))",
        "POLYGON ((0 0, 1 1, 1 0, 0 1, 0 0))",
    ]
    gseries = gpd.GeoSeries.from_wkt(geometries)

    bf_gseries = bpd.GeoSeries(gseries, session=session)

    result = bf_gseries.is_valid.to_pandas()
    expected = pd.Series([True, False], dtype="boolean", name="is_valid")

    pd.testing.assert_series_equal(expected, result, check_index=False)


def test_geoseries_is_simple(polars_session):
    session = polars_session
    geometries = [
        "LINESTRING (0 0, 1 1)",
        "LINESTRING (0 0, 1 1, 0 1, 1 0)",
    ]
    gseries = gpd.GeoSeries.from_wkt(geometries)

    bf_gseries = bpd.GeoSeries(gseries, session=session)

    result = bf_gseries.is_simple.to_pandas()
    expected = pd.Series([True, False], dtype="boolean", name="is_simple")

    pd.testing.assert_series_equal(expected, result, check_index=False)


def test_geoseries_is_ring(polars_session):
    session = polars_session
    geometries = [
        "LINESTRING (0 0, 1 0, 1 1, 0 1, 0 0)",
        "LINESTRING (0 0, 1 1, 1 0, 0 1)",
    ]
    gseries = gpd.GeoSeries.from_wkt(geometries)

    bf_gseries = bpd.GeoSeries(gseries, session=session)

    result = bf_gseries.is_ring.to_pandas()
    expected = pd.Series([True, False], dtype="boolean", name="is_ring")

    pd.testing.assert_series_equal(expected, result, check_index=False)


def test_geoseries_geom_type(polars_session):
    session = polars_session
    geometries = [
        "POINT (0 0)",
        "POLYGON ((0 0, 1 1, 0 1, 0 0))",
    ]
    gseries = gpd.GeoSeries.from_wkt(geometries)

    bf_gseries = bpd.GeoSeries(gseries, session=session)

    result = bf_gseries.geom_type.to_pandas()
    expected = pd.Series(
        ["ST_POINT", "ST_POLYGON"], dtype="string[pyarrow]", name="geom_type"
    )

    pd.testing.assert_series_equal(expected, result, check_index=False)