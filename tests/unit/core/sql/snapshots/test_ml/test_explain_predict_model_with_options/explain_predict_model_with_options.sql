SELECT * FROM ML.EXPLAIN_PREDICT(MODEL `my_model`, (SELECT * FROM new_data), OPTIONS(top_k_features = 5))
