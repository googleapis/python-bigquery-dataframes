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


def test_quickstart() -> None:
    import bigframes.pandas

    # We need a fresh session since we're modifying connection options.
    bigframes.pandas.close_session()

    # [START bigquery_bigframes_ordering_mode_partial]
    import bigframes.pandas as bpd

    bpd.options.bigquery.ordering_mode = "partial"
    # [END bigquery_bigframes_ordering_mode_partial]

    # [START bigquery_bigframes_ordering_mode_partial_ambiguous_window_warning]
    import warnings

    import bigframes.exceptions

    warnings.simplefilter(
        "ignore", category=bigframes.exceptions.AmbiguousWindowWarning
    )
    # [END bigquery_bigframes_ordering_mode_partial_ambiguous_window_warning]

    # Below is a copy of the main quickstart to check that it also works with
    # this ordering mode.

    # Create a DataFrame from a BigQuery table
    query_or_table = "bigquery-public-data.ml_datasets.penguins"
    df = bpd.read_gbq(query_or_table)

    # Use the DataFrame just as you would a pandas DataFrame, but calculations
    # happen in the BigQuery query engine instead of the local system.
    average_body_mass = df["body_mass_g"].mean()
    print(f"average_body_mass: {average_body_mass}")

    # Create the Linear Regression model
    from bigframes.ml.linear_model import LinearRegression

    # Filter down to the data we want to analyze
    adelie_data = df[df.species == "Adelie Penguin (Pygoscelis adeliae)"]

    # Drop the columns we don't care about
    adelie_data = adelie_data.drop(columns=["species"])

    # Drop rows with nulls to get our training data
    training_data = adelie_data.dropna()

    # Pick feature columns and label column
    X = training_data[
        [
            "island",
            "culmen_length_mm",
            "culmen_depth_mm",
            "flipper_length_mm",
            "sex",
        ]
    ]
    y = training_data[["body_mass_g"]]

    model = LinearRegression(fit_intercept=False)
    model.fit(X, y)
    model.score(X, y)

    assert model is not None
