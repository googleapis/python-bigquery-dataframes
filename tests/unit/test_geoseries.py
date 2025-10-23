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

import geopandas as gpd
import geopandas.testing
import pandas as pd

import bigframes.geopandas as bpd


def test_geoseries_is_empty(polars_session):
    session = polars_session
    gseries = gpd.GeoSeries(
        [
            gpd.points_from_xy([0], [0])[0],
            gpd.GeoSeries.from_wkt(["POLYGON EMPTY"])[0],
        ]
    )

    bf_gseries = bpd.GeoSeries(gseries, session=session)

    result = bf_gseries.is_empty.to_pandas()
    expected = gseries.is_empty

    pd.testing.assert_series_equal(
        expected, result, check_index=False, check_names=False, check_dtype=False
    )


def test_geoseries_is_valid(polars_session):
    session = polars_session
    gseries = gpd.GeoSeries.from_wkt(
        [
            "POLYGON ((0 0, 1 1, 0 1, 0 0))",
            "POLYGON ((0 0, 1 1, 1 0, 0 1, 0 0))",
        ]
    )

    bf_gseries = bpd.GeoSeries(gseries, session=session)

    result = bf_gseries.is_valid.to_pandas()
    expected = gseries.is_valid

    pd.testing.assert_series_equal(
        expected, result, check_index=False, check_names=False, check_dtype=False
    )


def test_geoseries_is_ring(polars_session):
    session = polars_session
    gseries = gpd.GeoSeries.from_wkt(
        [
            "LINESTRING (0 0, 1 0, 1 1, 0 1, 0 0)",
            "LINESTRING (0 0, 1 1, 1 0, 0 1)",
        ]
    )

    bf_gseries = bpd.GeoSeries(gseries, session=session)

    result = bf_gseries.is_ring.to_pandas()
    expected = gseries.is_ring

    pd.testing.assert_series_equal(
        expected, result, check_index=False, check_names=False, check_dtype=False
    )


def test_geoseries_is_simple(polars_session):
    session = polars_session
    gseries = gpd.GeoSeries.from_wkt(
        [
            "LINESTRING (0 0, 1 1)",
            "LINESTRING (0 0, 1 1, 0 1, 1 0)",
        ]
    )

    bf_gseries = bpd.GeoSeries(gseries, session=session)

    result = bf_gseries.is_simple.to_pandas()
    expected = gseries.is_simple

    pd.testing.assert_series_equal(
        expected, result, check_index=False, check_names=False, check_dtype=False
    )


def test_geoseries_geom_type(polars_session):
    session = polars_session
    gseries = gpd.GeoSeries.from_wkt(
        [
            "POINT (0 0)",
            "POLYGON ((0 0, 1 1, 0 1, 0 0))",
        ]
    )

    bf_gseries = bpd.GeoSeries(gseries, session=session)

    result = bf_gseries.geom_type.to_pandas()
    expected = gseries.geom_type

    pd.testing.assert_series_equal(
        expected, result, check_index=False, check_names=False, check_dtype=False
    )


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

    bf_gseries1 = bpd.GeoSeries(gseries1, session=session)
    bf_gseries2 = bpd.GeoSeries(gseries2, session=session)

    result = bf_gseries1.union(bf_gseries2).to_pandas().reset_index(drop=True)
    expected = gseries1.union(gseries2).reset_index(drop=True)

    gpd.testing.assert_geoseries_equal(result, expected, check_series_type=False)
