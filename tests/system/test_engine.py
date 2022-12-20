def test_read_gbq(engine, table_id_scalars, schema_scalars):
    df = engine.read_gbq(table_id_scalars)
    # TODO(swast): Test against public properties like columns or dtypes. Also,
    # check the names and data types match up.
    assert len(df._table.schema()) == len(schema_scalars)
