SELECT * FROM ML.PREDICT(MODEL `my_model`, (SELECT * FROM new_data), OPTIONS(quantiles = [0.25, 0.75]))
