# Copyright 2024 Google LLC
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


def test_linear_regression() -> None:
    # [START bigquery_dataframes_bqml_linear_regression]
    from bigframes.ml.linear_model import LinearRegression
    import bigframes.pandas as bpd

    # Load data from BigQuery
    bq_df = bpd.read_gbq("bigquery-public-data.ml_datasets.penguins")

    # Drop rows with nulls to get training data
    # use new subset thing
    training_data = bq_df.dropna()

    # Specify your feature (or input) columns and the label (or output) column:
    # drop - keep all columns except body_mass_g
    feature_columns = training_data[
        ["island", "culmen_length_mm", "culmen_depth_mm", "flipper_length_mm", "sex"]
    ]
    label_columns = training_data[["body_mass_g"]]

    test_data = bq_df[bq_df.body_mass_g.isnull()]

    # Create the linear model
    model = LinearRegression()
    model.fit(feature_columns, label_columns)

    # Score the model
    score = model.score(feature_columns, label_columns)

    # Predict using the model
    result = model.predict(test_data)
    # [END bigquery_dataframes_bqml_linear_regression]
    assert test_data is not None
    assert feature_columns is not None
    assert label_columns is not None
    assert model is not None
    assert score is not None
    assert result is not None
