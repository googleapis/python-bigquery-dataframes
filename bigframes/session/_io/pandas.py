# Copyright 2023 Google LLC
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

from typing import Dict

import geopandas  # type: ignore
import pandas
import pyarrow  # type: ignore

import bigframes.constants


def arrow_to_pandas(arrow_table: pyarrow.Table, dtypes: Dict):
    if len(dtypes) != arrow_table.num_columns:
        raise ValueError(
            f"Number of types {len(dtypes)} doesn't match number of columns "
            f"{arrow_table.num_columns}. {bigframes.constants.FEEDBACK_LINK}"
        )

    serieses = {}
    for column_name, column in zip(arrow_table.column_names, arrow_table):
        dtype = dtypes[column_name]

        if dtype == geopandas.array.GeometryDtype():
            series = geopandas.GeoSeries.from_wkt(
                column,
                # BigQuery geography type is based on the WGS84 reference ellipsoid.
                crs="EPSG:4326",
            )
        else:
            series = pandas.Series(column, dtype=dtype)

        serieses[column_name] = series

    return pandas.DataFrame(serieses)
