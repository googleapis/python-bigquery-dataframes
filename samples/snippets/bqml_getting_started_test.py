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


def test_bqml_getting_started():

    import bigframes
    import bigframes.pandas as bpd

    #
    df = bpd.read_gbq(
        """
    SELECT GENERATE_UUID() AS rowindex, *
    FROM
    `bigquery-public-data.google_analytics_sample.ga_sessions_*`
    WHERE
    _TABLE_SUFFIX BETWEEN '20160801' AND '20170630'
    """,
        index_col="rowindex",
    )

    # make comments

    totals = df["totals"]

    # using totals, selecting id for transaction example
    totals["0000fb2c-2861-40be-9c6c-309afd7e7883"]

    transactions = totals.struct.field("transactions")

    label = transactions.notnull().map({True: 1, False: 0})

    operatingSystem = df["device"].struct.field("operatingSystem")

    operatingSystem = operatingSystem.fillna("")

    isMobile = df["device"].struct.field("isMobile")

    country = df["geoNetwork"].struct.field("country").fillna("")

    pageviews = totals.struct.field("pageviews").fillna(0)

    features = bpd.DataFrame(
        {"os": operatingSystem, "is_mobile": isMobile, "pageviews": pageviews}
    )

    # printing out the dataframe
    df

    from bigframes.ml.linear_model import LogisticRegression

    model = LogisticRegression()

    model.fit(features, label)
    model.to_gbq("bqml_tutorial.sample_model", replace=True)
