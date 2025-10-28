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

# Inspired by the SQL at https://cloud.google.com/blog/products/data-analytics/a-closer-look-at-earth-engine-in-bigquery

import typing

import bigframes.bigquery as bbq
import bigframes.pandas as bpd


def test_wildfire_risk(session):
    # Step 1: Select inputs from datasets that we've subscribed to
    wildfire_raster = bpd.read_gbq("wildfire_risk_to_community_v0_mosaic.fire")[
        "assets.image.href"
    ]
    places = bpd.read_gbq("bigquery-public-data.geo_us_census_places.places_colorado")[
        ["place_id", "place_name", "place_geom"]
    ]
    places = places.rename(columns={"place_geom": "geo"})

    # Step 2: Compute the weather forecast using WeatherNext Graph forecast data
    weather_forecast = bpd.read_gbq("weathernext_graph_forecasts.59572747_4_0")
    weather_forecast = weather_forecast[
        weather_forecast["init_time"] == "2025-04-28 00:00:00+00:00"
    ]
    weather_forecast = weather_forecast.explode("forecast")
    wind_speed = (
        weather_forecast["forecast"]["10m_u_component_of_wind"] ** 2
        + weather_forecast["forecast"]["10m_v_component_of_wind"] ** 2
    ) ** 0.5
    weather_forecast = weather_forecast.assign(wind_speed=wind_speed)
    weather_forecast = weather_forecast[weather_forecast["forecast"]["hours"] < 24]
    weather_forecast = typing.cast(
        bpd.DataFrame,
        weather_forecast.merge(
            places, how="inner", left_on="geography_polygon", right_on="geo"
        ),
    )
    weather_forecast = weather_forecast.groupby("place_id").agg(
        place_name=("place_name", "first"),
        geo=("geo", "first"),
        average_wind_speed=("wind_speed", "mean"),
        maximum_wind_speed=("wind_speed", "max"),
    )

    # Step 3: Combine with wildfire risk for each community
    wildfire_risk = weather_forecast.assign(
        wildfire_likelihood=bbq.st_regionstats(
            weather_forecast["geo"],
            wildfire_raster,
            "BP",
            options={"scale": 1000},
        )["mean"],
        wildfire_consequence=bbq.st_regionstats(
            weather_forecast["geo"],
            wildfire_raster,
            "CRPS",
            options={"scale": 1000},
        )["mean"],
    )

    # Step 4: Compute a simple composite index of relative wildfire risk.
    relative_risk = (
        wildfire_risk["wildfire_likelihood"].rank(pct=True)
        + wildfire_risk["wildfire_consequence"].rank(pct=True)
        + wildfire_risk["average_wind_speed"].rank(pct=True)
    ) / 3 * 100
    wildfire_risk = wildfire_risk.assign(relative_risk=relative_risk)
    assert wildfire_risk is not None
