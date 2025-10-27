def test_to_pandas_batches_with_json_columns(session):
    """Test that JSON columns are properly handled in to_pandas_batches."""
    # Create a DataFrame with JSON column
    df = session.read_gbq('SELECT JSON \'{"key": "value"}\' as json_col')

    # This should not raise an error
    batches = df._to_pandas_batches(page_size=10)
    result = next(batches)

    # Verify the result is a string representation
    assert isinstance(result["json_col"].iloc[0], str)
