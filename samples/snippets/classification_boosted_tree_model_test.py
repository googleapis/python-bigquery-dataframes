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


def test_boosted_tree_model(random_model_id: str) -> None:
    your_model_id = random_model_id
    # [START bigquery_dataframes_bqml_boosted_tree_prepare]
    import bigframes.pandas as bpd

    input_data = bpd.read_gbq(
        "bigquery-public-data.ml_datasets.census_adult_income",
        columns=(
            "age",
            "workclass",
            "marital_status",
            "education_num",
            "occupation",
            "hours_per_week",
            "income_bracket",
            "functional_weight",
        ),
    )
    input_data["dataframe"] = bpd.Series("training", index=input_data.index,).case_when(
        [
            (((input_data["functional_weight"] % 10) == 8), "evaluation"),
            (((input_data["functional_weight"] % 10) == 9), "prediction"),
        ]
    )
    del input_data["functional_weight"]
    # [END bigquery_dataframes_bqml_boosted_tree_prepare]
    # [START bigquery_dataframes_bqml_boosted_tree_predict]
    # Select model you'll use for predictions. `read_gbq_model` loads model
    # data from BigQuery, but you could also use the `tree_model` object
    # from previous steps.
    tree_model = bpd.read_gbq_model(
        your_model_id,  # For example: "your-project.bqml_tutorial.tree_model"
    )

    # input_data is defined in an earlier step.
    prediction_data = input_data[input_data["dataframe"] == "prediction"]

    predictions = tree_model.predict(prediction_data)
    predictions.peek()
    # Output:
    # predicted_income_bracket   predicted_income_bracket_probs.label  predicted_income_bracket_probs.prob
    #                   <=50K                                   >50K                   0.05183430016040802                 |
    #                                                           <50K                   0.94816571474075317
    #                   <=50K                                   >50K                   0.00365859130397439
    #                                                           <50K                   0.99634140729904175
    #                   <=50K                                   >50K                   0.037775970995426178
    #                                                           <50K                   0.96222406625747681
    # [END bigquery_dataframes_bqml_boosted_tree_predict]
    assert input_data is not None
    assert tree_model is not None
    assert predictions is not None
    assert prediction_data is not None
