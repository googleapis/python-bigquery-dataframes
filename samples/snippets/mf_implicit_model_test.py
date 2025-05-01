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


def test_implicit_matrix_factorization(random_model_id: str) -> None:
    # [START bigframes_dataframe_mf_implicit_data]
    from bigframes.ml import decomposition
    import bigframes.pandas as bpd

    # sample data must be created from joined data and then grouped and ordered
    bq_df = bpd.read_gbq("bqml_tutorial.analytics_session_data")
    print(bq_df.peek(5))
    # Expected output:
    #
    # [END bigframes_dataframe_mf_implicit_data]
    # [START bigframes_dataframe_mf_implicit_model]
    rating_calculation = 0.3 * (1 + (bq_df["session_duration"] - 57937) / 57937)
    filtered_bq_df = bq_df[rating_calculation < 1].assign(
        rating=rating_calculation[rating_calculation < 1]
    )
    model = decomposition.MatrixFactorization(
        num_factors=15,
        feedback_type="implicit",
        user_col="visitorId",
        item_col="contentId",
        rating_col="rating",
        l2_reg=30,
    )
    model.fit(filtered_bq_df)
    # [END bigframes_dataframe_mf_implicit_model]
    # [START bigframes_dataframe_mf_implicit_evaluate]
    model.score()
    # Output:
    # [END bigframes_dataframe_mf_implicit_evaluate]
    # [START bigframes_dataframe_mf_implicit_subset]
    # [END bigframes_dataframe_mf_implicit_subset]
    # [START bigframes_dataframe_mf_implicit_recommend]
    # [END bigframes_dataframe_mf_implicit_recommend]
    pass
