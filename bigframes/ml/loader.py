from __future__ import annotations

from typing import TYPE_CHECKING

from google.cloud import bigquery

if TYPE_CHECKING:
    import bigframes

import bigframes.ml.linear_model


def from_bq(
    session: bigframes.Session, model: bigquery.Model
) -> bigframes.ml.linear_model.LinearRegression:
    if model.model_type == "LINEAR_REGRESSION":
        return bigframes.ml.linear_model.LinearRegression._from_bq(session, model)
    else:
        raise NotImplementedError(
            f"Model type {model.model_type} is not yet supported by BigFrames"
        )
