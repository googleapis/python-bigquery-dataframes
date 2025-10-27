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

import pytest

import bigframes.bigquery as bbq
import bigframes.geopandas as gpd
import bigframes.pandas as bpd

pytest.importorskip("pytest_snapshot")


def test_st_regionstats(compiler_session, snapshot):
    geos = gpd.GeoSeries(["POINT(1 1)"], session=compiler_session)
    rasters = bpd.Series(["raster_uri"], dtype="string", session=compiler_session)
    df = bbq.st_regionstats(geos, rasters, "band1", {"scale": 100})
    assert "area" in df.columns
    snapshot.assert_match(df.sql, "out.sql")
