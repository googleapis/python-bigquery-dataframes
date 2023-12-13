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

def test_kmeans_sample():
# [START bigquery_dataframes_bqml_kmeans]
    import bigframes.pandas as bpd
    import bigframes
    from bigframes import dataframe
    import bigframes.pandas as bpd
    import datetime

    #Load data from BigQuery
    h = bpd.read_gbq("bigquery-public-data.london_bicycles.cycle_hire",  h.rename(
        columns = {"start_station_name": "station_name", "start_station_id": "station_id"}
    ))
    s = bpd.read_gbq(
        """
        SELECT
        id,
        ST_DISTANCE(
            ST_GEOGPOINT(s.longitude, s.latitude),
            ST_GEOGPOINT(-0.1, 51.5)
        ) / 1000 AS distance_from_city_center
        FROM
        `bigquery-public-data.london_bicycles.cycle_stations` s
        """ )

    # transform data into queryable format
    sample_time = datetime.datetime(2015, 1, 1, 0, 0, 0, tzinfo=datetime.timezone.utc)
    sample_time2 = datetime.datetime(2016, 1, 1, 0, 0, 0, tzinfo=datetime.timezone.utc)

    h = h.loc[(h["start_date"] >= sample_time) & (h["start_date"] <= sample_time2)]

    h.start_date.dt.dayofweek.map(
        {
            0: "weekday",
            1: "weekday",
            2: "weekday",
            3: "weekday",
            4: "weekday",
            5: "weekend",
            6: "weekend",
        }
    )

    #merge dataframes h and s
    merged_df = h.merge(
        right=s,
        how="inner",
        left_on="station_id",
        right_on="id",
    )
    # Create new dataframe variable from merge: 'stationstats' 
    stationstats = merged_df.groupby("station_name").agg(
        {"duration": ["mean", "count"], "distance_from_city_center": "max"}
    )
    # [END bigquery_dataframes_bqml_kmeans]
    

    # [START bigquery_dataframes_bqml_kmeans_fit]

    # import the KMeans model from bigframes.ml to cluster the data
    from bigframes.ml.cluster import KMeans

    cluster_model = KMeans(n_clusters=4)
    cluster_model = cluster_model.fit(stationstats).to_gbq(cluster_model)

    # [END bigquery_dataframes_bqml_kmeans_fit]
    
    # [START bigquery_dataframes_bqml_kmeans_predict]

    # Use 'contains' function to find all entries with string "Kennington". 
    stationstats = stationstats.str.contains("Kennington")

    #Predict using the model
    result = cluster_model.predict(stationstats)

    # [END bigquery_dataframes_bqml_kmeans_predict]

    assert result is not None
