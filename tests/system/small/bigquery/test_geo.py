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
from shapely.geometry import LineString, Point, Polygon  # type: ignore

import bigframes.bigquery as bbq
import bigframes.geopandas
import bigframes.series


def test_geo_st_area():
    data = [
        Polygon([(0.0, 0.0), (0.1, 0.1), (0.0, 0.1)]),
        Polygon([(0.10, 0.4), (0.9, 0.5), (0.10, 0.5)]),
        Polygon([(0.1, 0.1), (0.2, 0.1), (0.2, 0.2)]),
        LineString([(0, 0), (1, 1), (0, 1)]),
        Point(0, 1),
    ]

    geopd_s = geopandas.GeoSeries(data=data, crs="EPSG:4326")
    geobf_s = bigframes.geopandas.GeoSeries(data=data)

    # Round both results to get an approximately similar output
    geopd_s_result = geopd_s.to_crs(26393).area.round(-7)
    geobf_s_result = bbq.st_area(geobf_s).to_pandas().round(-7)

    pd.testing.assert_series_equal(
        geobf_s_result,
        geopd_s_result,
        check_dtype=False,
        check_index_type=False,
        check_exact=False,
        rtol=1,
    )
