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
    # [START bigquery_getting_Started_bqml_tutorial]
    from bigframes.ml.linear_model import LogisticRegression
    import bigframes.pandas as bpd

    # Start by selecting the data you'll use for training. `read_gbq` accepts
    # either a SQL query or a table ID. Since this example selects from multiple
    # tables via a wildcard, use SQL to define this data. Watch issue
    # https://github.com/googleapis/python-bigquery-dataframes/issues/169
    # for updates to `read_gbq` to support wildcard tables.
    #
    # https://github.com/googleapis/python-bigquery-dataframes/issues/169

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

    # Extract the total number of transactions within
    # the Google Analytics session.
    #
    # Because the totals column is a STRUCT data type, we need to call
    # Series.struct.field("transactions") to extract the transactions field.
    # See the reference documentation below:
    # https://cloud.google.com/python/docs/reference/bigframes/latest/bigframes.operations.structs.StructAccessor#bigframes_operations_structs_StructAccessor_field
    transactions = df["totals"].struct.field("transactions")

    # If the number of transactions is NULL, the value in the label
    # column is set to 0. Otherwise, it is set to 1. These values
    # represent the possible outcomes.
    label = transactions.notnull().map({True: 1, False: 0})

    # Choosing the operating system of the users devices.
    operatingSystem = df["device"].struct.field("operatingSystem")
    operatingSystem = operatingSystem.fillna("")

    # Extract whether the visitor's device is a mobile device.
    isMobile = df["device"].struct.field("isMobile")

    # Extract where the visitors country of origin is.
    country = df["geoNetwork"].struct.field("country").fillna("")

    # Total number of pageviews within the session.
    pageviews = df["totals"].struct.field("pageviews").fillna(0)

    # Selecting values to represent data in columns in DataFrames.
    features = bpd.DataFrame(
        {
            "os": operatingSystem,
            "is_mobile": isMobile,
            "country": country,
            "pageviews": pageviews,
        }
    )

    # Logistic Regression model splits data into two classes, giving the
    # probablity the data is in one of the classes.
    model = LogisticRegression()
    model.fit(features, label)

    # When writing a DataFrame to a BigQuery table, include destinaton table
    # and parameters, index defaults to "True".
    model.to_gbq("bqml_tutorial.sample_model", replace=True)
    # [END bigquery_getting_Started_bqml_tutorial]
