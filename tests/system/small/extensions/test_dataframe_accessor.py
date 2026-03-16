import pandas as pd
import pytest

import bigframes.pandas as bpd

def test_pandas_ai_forecast(session):
    df = pd.DataFrame(
        {
            "parsed_date": pd.to_datetime(
                ["2017-01-01", "2017-01-02", "2017-01-03", "2017-01-04", "2017-01-05"]
            ),
            "total_visits": [10.0, 20.0, 30.0, 40.0, 50.0],
            "id": ["1", "1", "1", "1", "1"]
        }
    )

    result = df.bigquery.ai.forecast(
        timestamp_col="parsed_date",
        data_col="total_visits",
        horizon=1,
        session=session,
    )

    assert "forecast_timestamp" in result.columns
    assert "forecast_value" in result.columns
