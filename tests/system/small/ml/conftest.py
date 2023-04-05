from typing import cast

import pytest

import bigframes.ml.core
import bigframes.ml.linear_model


@pytest.fixture(scope="session")
def penguins_bqml_model(session, penguins_model_name) -> bigframes.ml.core.BqmlModel:
    model = session.bqclient.get_model(penguins_model_name)
    return bigframes.ml.core.BqmlModel(session, model)


@pytest.fixture(scope="session")
def penguins_model_loaded(
    session, penguins_model_name
) -> bigframes.ml.linear_model.LinearRegression:
    return cast(
        bigframes.ml.linear_model.LinearRegression,
        session.read_gbq_model(penguins_model_name),
    )
