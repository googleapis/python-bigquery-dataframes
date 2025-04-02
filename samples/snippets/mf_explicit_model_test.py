# Copyright 2024 Google LLC
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


def test_explicit_matrix_factorization(random_model_id: str) -> None:
    your_model_id = random_model_id
    # [START bigframes_dataframes_bqml_mf_explicit_create]
    from bigframes.ml import decomposition
    import bigframes.pandas as bpd

    # Load data from BigQuery
    bq_df = bpd.read_gbq(
        "bqml_tutorial.ratings", columns=("user_id", "item_id", "rating")
    )

    # Create the Matrix Factorization model
    model = decomposition.MatrixFactorization(
        num_factors=34,
        feedback_type="explicit",
        user_col="user_id",
        item_col="item_id",
        rating_col="rating",
        l2_reg=9.83,
    )
    model.fit(bq_df)
    model.to_gbq(
        your_model_id, replace=True  # For example: "bqml_tutorial.mf_explicit"
    )
    # [END bigframes_dataframes_bqml_mf_explicit_create]
    # [START bigframes_dataframe_bqml_mf_explicit_evaluate]
    import bigframes.pandas as bpd

    model.score(bq_df)
    # [END bigframes_dataframe_bqml_mf_explicit_evaluate]
    # [START bigframes_dataframe_bqml_mf_explicit_predict]

    # [END bigframes_dataframe_bqml_mf_explicit_predict]
    # [START bigframes_dataframe_bqml_mf_explicit_recommend]
    model.predict(bq_df)
    # [END bigframes_dataframe_bqml_mf_explicit_recommend]
    pass
