import bigframes.ml.sql as ml_sql


def test_create_model_produces_correct_sql():
    sql = ml_sql.create_model(
        model_name="my_dataset.my_model",
        source_sql="SELECT * FROM my_table",
        options={"model_type": "lin_reg", "input_label_cols": ["col_a"], "l1_reg": 0.6},
    )
    assert (
        sql
        == """CREATE MODEL `my_dataset.my_model`
OPTIONS (
  model_type="lin_reg",
  input_label_cols=["col_a"],
  l1_reg=0.6
) AS SELECT * FROM my_table"""
    )


def test_ml_predict_produces_correct_sql():
    sql = ml_sql.ml_predict(
        model_name="my_dataset.my_model", source_sql="SELECT * FROM my_table"
    )
    assert (
        sql
        == """SELECT * FROM ML.PREDICT(MODEL `my_dataset.my_model`,
  (SELECT * FROM my_table))"""
    )


def test_ml_evaluate_produces_correct_sql():
    sql = ml_sql.ml_evaluate(
        model_name="my_dataset.my_model", source_sql="SELECT * FROM my_table"
    )
    assert (
        sql
        == """SELECT * FROM ML.EVALUATE(MODEL `my_dataset.my_model`,
  (SELECT * FROM my_table))"""
    )


def test_ml_evaluate_no_source_produces_correct_sql():
    sql = ml_sql.ml_evaluate(model_name="my_dataset.my_model")
    assert sql == """SELECT * FROM ML.EVALUATE(MODEL `my_dataset.my_model`)"""
