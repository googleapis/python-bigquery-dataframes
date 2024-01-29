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
    df = bpd.read_gbq('''
        SELECT PARSE_TIMESTAMP("%Y%m%d", date) AS parsed_date,
        SUM(totals.visits) AS total_visits
        FROM
        `bigquery-public-data.google_analytics_sample.ga_sessions_*`
        GROUP BY date
        ''')
    X = df[["parsed_date"]]
    y = df[["total_visits"]]

    # Create an Arima-based time series model using the Google Analytics 360 data.
    from bigframes.ml.forecasting import ARIMAPlus

    ga_arima_model = ARIMAPlus()

    # Fit the model to your dataframe.
    ga_arima_model.fit(X,y)

    # The model.fit() call above created a temporary model.
    # Use the to_gbq() method to write to a permanent location.
    ga_arima_model.to_gbq(
    your_model_id,  # For example: "bqml_tutorial.sample_model",
    replace=True,
    )

    # Inspect the evaluation metrics of all evaluated models.
    # when ruuning this function use same model, dataset, model name (str)
    evaluation = ga_arima_model.summary(
        f'''
        SELECT *   
        FROM ML.ARIMA_EVALUATE(MODEL `{your_model_id}`)
        '''
        )
    
    print(evaluation)
    # Inspect the coefficients of your model
    
    