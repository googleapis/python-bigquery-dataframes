from bigframes.testing import mocks


def test_clear_api_methods_log():
    """
    This test exists to provoke a race condition in other tests.
    It makes a simple query which has the side effect of clearing
    the global _api_methods list in the log_adapter.
    """
    session = mocks.create_bigquery_session()
    _ = session.bqclient.query("SELECT 1")
