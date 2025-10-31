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


def test_st_regionstats():
    project_id = "bigframes-dev"

    # [START bigquery_dataframes_st_regionstats]
    from typing import cast

    import bigframes.bigquery as bq
    import bigframes.pandas as bpd

    # TODO: Set the project_id to your Google Cloud project ID.
    # project_id = "your-project-id"
    #
    # TODO: Set the dataset_id to the ID of the dataset that contains the
    # `climate` table. This is likely a linked dataset to Earth Engine.
    # See: https://cloud.google.com/bigquery/docs/link-earth-engine
    linked_dataset = "era5_land_daily_aggregated"

    # Load the table of country boundaries.
    bpd.options.bigquery.project = project_id
    countries = bpd.read_gbq("bigquery-public-data.overture_maps.division_area")

    # Filter to just the countries.
    countries = countries[countries["subtype"] == "country"].copy()
    countries["name"] = countries["names"].struct.field("primary")

    # TODO: Add st_simplify when it is available in BigFrames.
    # https://github.com/googleapis/python-bigquery-dataframes/issues/1497
    # countries["simplified_geometry"] = bq.st_simplify(countries["geometry"], 10000)
    countries["simplified_geometry"] = countries["geometry"]

    # Get the reference to the temperature data from a linked dataset.
    # Note: This sample assumes you have a linked dataset to Earth Engine.
    # See: https://cloud.google.com/bigquery/docs/link-earth-engine
    image_href = bpd.read_gbq(f"{project_id}.{linked_dataset}.climate").where(
        lambda df: df["start_datetime"] == "2025-01-01 00:00:00"
    )
    raster_id = image_href["assets"].struct.field("image").struct.field("href").item
    stats = bq.st_regionstats(
        countries["simplified_geometry"],
        raster_id=cast(str, raster_id),
        band="temperature_2m",
    )

    # Extract the mean and convert from Kelvin to Celsius.
    countries["mean_temperature"] = stats.struct.field("mean") - 273.15

    # Sort by the mean temperature to find the warmest countries.
    result = countries[["name", "mean_temperature"]].sort_values(
        "mean_temperature", ascending=False
    )
    print(result.head())
    # [END bigquery_dataframes_st_regionstats]
