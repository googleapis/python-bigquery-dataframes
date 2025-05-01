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
    # Evaluate the model using the score() function
    model.score()
    # Output:
    # mean_absolute_error	mean_squared_error	mean_squared_log_error	median_absolute_error	r2_score	explained_variance
    # 0.485403	                0.395052	        0.025515	            0.390573	        0.68343	        0.68343
    # [END bigframes_dataframe_bqml_mf_explicit_evaluate]
    # [START bigframes_dataframe_bqml_mf_recommend_df]
    subset = bq_df.head(6)
    predicted = model.predict(subset)
    print(predicted)
    # Output:
    #   predicted_rating	user_id	 item_id	rating
    # 0	    4.206146	     4354	  968	     4.0
    # 1	    4.853099	     3622	  3521	     5.0
    # 2	    2.679067	     5543	  920	     2.0
    # 3	    4.323458	     445	  3175	     5.0
    # 4	    3.476911	     5535	  235	     4.0
    # [END bigframes_dataframe_bqml_mf_explicit_recommend_df]
    # [START bigframes_dataframe_bqml_mf_explicit_recommend_model]
    # import bigframes.bigquery as bbq

    # TODO: implement right_index parameter for DataFrame.merge()
    # # Load movie data from BigQuery
    # movies = bpd.read_gbq("bqml_tutorial.movies")
    # # Merge movie data with rating data
    # merged_df = bpd.merge(predicted, movies, left_on='item_id', right_on='movie_id')
    # # separate users from data to call struct on data
    # users = merged_df[['user_id', 'item_id']]
    # user_data = merged_df[['movie_title', 'genre', 'predicted_rating', 'movie_id']].set_index('movie_id')
    # struct_data = bbq.struct(user_data).to_frame()
    # # Merge data to groupby predicted_rating and sort
    # merged_user = bpd.merge(users, struct_data, left_on='item_id', right_index=True).drop('item_id', axis=1)
    # desc_pred = merged_user.sort_values(by='predicted_rating', ascending=False)
    # grouped = desc_pred.groupby('predicted_rating')
    # result = bbq.array_agg(grouped)
    # result.head(5)
    # Output:
    # [END bigframes_dataframe_bqml_mf_explicit_recommend_model]
    pass
