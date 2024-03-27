def test_multiple_timeseries_forecasting_model(random_model_id):
    # [START bigquery_dataframes_bqml_create_data__set(1)]
    your_model_id = random_model_id

    from bigframes.ml import forecasting
    import bigframes.pandas as bpd

    # Start by selecting the data you'll use for training. `read_gbq_table` accepts
    # either a SQL query or a table ID. Since this example selects from multiple
    # tables via a wildcard, use SQL to define this data. Watch issue
    # https://github.com/googleapis/python-bigquery-dataframes/issues/169
    # for updates to `read_gbq_table` to support wildcard tables.

    df = bpd.read_gbq_table("bigquery-public-data.new_york.citibike_trips", filters=[])

    # [END bigquery_dataframes_bqml_create_data__set(1)]

    # [START bigquery_dataframes_bqml_visualize_time_series_to_forecast(2)]
    features = bpd.DataFrame(
        {
            "num_trips": df.starttime,
            "date": df["starttime"].dt.date,
        }
    )
    date = df["starttime"].dt.date
    df.groupby([date])
    num_trips = features.groupby(["date"]).count()

    # Results from running "print(num_trips)"

    #                num_trips
    # date
    # 2013-07-01      16650
    # 2013-07-02      22745
    # 2013-07-03      21864
    # 2013-07-04      22326
    # 2013-07-05      21842
    # 2013-07-06      20467
    # 2013-07-07      20477
    # 2013-07-08      21615
    # 2013-07-09      26641
    # 2013-07-10      25732
    # 2013-07-11      24417
    # 2013-07-12      19006
    # 2013-07-13      26119
    # 2013-07-14      29287
    # 2013-07-15      28069
    # 2013-07-16      29842
    # 2013-07-17      30550
    # 2013-07-18      28869
    # 2013-07-19      26591
    # 2013-07-20      25278
    # 2013-07-21      30297
    # 2013-07-22      25979
    # 2013-07-23      32376
    # 2013-07-24      35271
    # 2013-07-25      31084

    # LINE GRAPH GOES HERE
    # EXPLAIN WHY WER
    num_trips.plot.line(
        rot=45,
    )

    # [END bigquery_dataframes_bqml_visualize_time_series_to_forecast(2)]

    # [START bigquery_dataframes_bqml_visualize_time_series_to_forecast(3)]

    date = df["starttime"].dt.date
    df.groupby([date])
    # EXPLAIN AS INDEX
    num_trips = features.groupby(["date"], as_index=False).count()
    features = bpd.DataFrame(
        {
            "num_trips": df.starttime,
            "date": df["starttime"].dt.date,
        }
    )

    model = forecasting.ARIMAPlus()

    X = num_trips["date"].to_frame()
    y = num_trips["num_trips"].to_frame()

    model.fit(X, y)
    # The model.fit() call above created a temporary model.
    # Use the to_gbq() method to write to a permanent location.

    your_model_id = "stabd-testing.bqml_tutorial.nyc_citibike_arima_model"

    model.to_gbq(
        your_model_id,  # For example: "bqml_tutorial.sample_model",
        replace=True,
    )  # RESULTS OF MODEL FIT:
    # ARIMAPlus : (auto_arima_max_order=5, data_frequency='AUTO_FREQUENCY',
    # max_time_series_length=3, min_time_series_length=20,
    # time_series_length_fraction=1.0, trend_smoothing_window_size=-1)

    # [END bigquery_dataframes_bqml_visualize_time_series_to_forecast(3)]

    # [START bigquery_dataframes_bqml_visualize_time_series_to_forecast(4)]


#  model.summary()
# non_seasonal_p	non_seasonal_d	non_seasonal_q	has_drift	log_likelihood	AIC	variance	seasonal_periods	has_holiday_effect	has_spikes_and_dips	has_step_changes	error_message
#  0	0	1	5	False	-11291.255555	22594.51111	10665799.388004	['WEEKLY' 'YEARLY']	False	True	True
#  1 rows × 12 columns

# [1 rows x 12 columns in total]    # [END bigquery_dataframes_bqml_visualize_time_series_to_forecast(4)]
