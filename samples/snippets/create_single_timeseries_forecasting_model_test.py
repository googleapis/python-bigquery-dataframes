# Copyright 2023 Google LLC
#
# Licensed under the Apache License, Version 2.0 (t
# you may not use this file except in compliance wi
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in
# distributed under the License is distributed on a
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, eit
# See the License for the specific language governi
# limitations under the License.


def test_create_single_timeseries(random_model_id):
    your_model_id = random_model_id

    # [START bigquery_dataframes_single_timeseries_forecasting_model_tutorial]
    import bigframes.pandas as bpd
    
    # Start by selecting the data you'll use for training. `read_gbq` accepts
    # either a SQL query or a table ID. Since this example selects from multiple
    # tables via a wildcard, use SQL to define this data. Watch issue
    # https://github.com/googleapis/python-bigquery-dataframes/issues/169
    # for updates to `read_gbq` to support wildcard tables.
    
    # Read and visualize the time series you want to forecast.
    df = bpd.read_gbq(
        'bigquery-public-data.google_analytics_sample.ga_sessions_*'
        )
    parsed_date = bpd.to_datetime(df.date, format= "%Y%m%d", utc = True)
    total_visits = df.groupby(["date"])["parsed_date"].sum()
    visits = df["totals"].struct.field("visits")

    # Create an Arima-based time series model using the Google Analytics 360 data. 
    from bigframes.ml.forecasting import ARIMAPlus

    ga_arima_model = ARIMAPlus()

    X = df[["parsed_date"]]
    y = df[["total_visits"]]

    # Fit the model to your dataframe.
    ga_arima_model.fit(X,y)

    # The model.fit() call above created a temporary model.
    # Use the to_gbq() method to write to a permanent location.
    ga_arima_model.to_gbq(
    your_model_id,  # For example: "bqml_tutorial.sample_model",
    replace=True,
    )

    # Inspect the evaluation metrics of all evaluated models.
    # when running this function use same model, dataset, model name (str)
    evaluation = ga_arima_model.summary(
        show_all_candidate_models = False,
        )
    
    print(evaluation)

    # Inspect the coefficients of your model
    f'''
    SELECT *
    FROM ML.ARIMA_COEFFICIENTS(MODEL `{your_model_id}`)
    '''
    evaluation.ML.ARIMA_COEFFICIENTS()

    # Use your model to forecast the time series
    #standardSQL
    your_model_id.forecast()

    # Explain and visualize the forecasting results
    f'''
    SELECT *
    FROM ML.EXPLAIN_FORECAST(
    MODEL `{your_model_id}`,
    STRUCT(
    [horizon AS horizon]
    [, confidence_level AS confidence_level]))
    '''
    total_visits.plot.line(x = 'history_timestamp', y = 'history_value')
    
    # Visualize the forecasting results without having decompose_time_series enabled.
    df = bpd.read_gbq(
    'bigquery-public-data.google_analytics_sample.ga_sessions_*'
    )
    parsed_date = bpd.to_datetime(df.date, format= "%Y%m%d", utc = True)
    visits = df["totals"].struct.field("visits")
    df = bpd.DataFrame(
    {
        'history_timestamp': parsed_date,
        'history_value': visits,
    }
    )
    total_visits = df.groupby(["history_timestamp"], as_index = False).sum(numeric_only= True)
