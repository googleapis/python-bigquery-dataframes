import hashlib
import logging

from google.cloud import bigquery
from google.cloud.exceptions import NotFound
import pytest

import bigframes


@pytest.fixture(scope="module")
def penguins_model(
    session: bigframes.Session, dataset_id_permanent, penguins_table_id
) -> bigquery.Model:
    """Provides a pretrained model as a test fixture that is cached across test runs.
    This lets us run system tests without having to wait for a model.fit(...)"""
    sql = """
CREATE OR REPLACE MODEL `$model_name`
OPTIONS
  (model_type='linear_reg',
  input_label_cols=['body_mass_g']) AS
SELECT
  *
FROM
  `$table_id`
WHERE
  body_mass_g IS NOT NULL"""
    # We use the SQL hash as the name to ensure the model is regenerated if this fixture is edited
    model_name = f"{dataset_id_permanent}.{hashlib.md5(sql.encode()).hexdigest()}"
    sql = sql.replace("$model_name", model_name)

    # TODO(bmil): move this to the original SQL construction once penguins_table has been
    # migrated to the permanent dataset too
    sql = sql.replace("$table_id", penguins_table_id)

    try:
        return session.bqclient.get_model(model_name)
    except NotFound:
        logging.info(
            "penguins_model fixture was not found in the permanent dataset, regenerating it..."
        )
        session.bqclient.query(sql).result()
        return session.bqclient.get_model(model_name)
