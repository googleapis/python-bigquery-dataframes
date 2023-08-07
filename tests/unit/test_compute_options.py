import bigframes as bf


def test_maximum_bytes_option(session: bf.Session, mock_bigquery_client):
    num_query_calls = 0
    with bf.option_context("compute.maximum_bytes_billed", 10000):
        # clear initial method calls
        mock_bigquery_client.method_calls = []
        session._start_query("query")
        for call in mock_bigquery_client.method_calls:
            name, _, kwargs = call
            if name == "query":
                num_query_calls = +1
                assert kwargs["job_config"].maximum_bytes_billed == 10000
    assert num_query_calls > 0
