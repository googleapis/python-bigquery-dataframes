"""Bigframes provides a DataFrame API for BigQuery."""

import copy
from typing import Optional

import google.auth.credentials
import google.cloud.bigquery
import ibis
import ibis.expr.types as ibis_types
import ibis_bigquery
import pandas


class Context:
  """Encapsulates configuration for working with an Engine.

  Attributes:
    credentials: The OAuth2 Credentials to use for this client. If not passed
      falls back to the default inferred from the environment.
    project: Project ID for the project which the client acts on behalf of. Will
      be passed when creating a dataset / job. If not passed, falls back to the
      default inferred from the environment.
    location: Default location for jobs / datasets / tables.
  """

  def __init__(
      self,
      credentials: Optional[google.auth.credentials.Credentials] = None,
      project: Optional[str] = None,
      location: Optional[str] = None):
    self._credentials = credentials
    self._project = project
    self._location = location

  @property
  def credentials(self) -> google.auth.credentials.Credentials:
    return self._credentials

  @property
  def project(self) -> str:
    return self._project

  @property
  def location(self) -> str:
    return self._location


class Engine:
  """A DataFrame interface to BigQuery."""

  def __init__(self, context: Optional[Context] = None):
    if context is not None:
      self.bqclient = google.cloud.bigquery.Client(
          credentials=context.credentials,
          project=context.project,
          location=context.location)
      self.ibis_client = ibis_bigquery.connect(
          project_id=context.project, credentials=context.credentials)
    else:
      self.bqclient = google.cloud.bigquery.Client()
      self.ibis_client = ibis_bigquery.connect(project_id=self.bqclient.project)

  def read_gbq(self, table: str) -> "DataFrame":
    """Loads data from Google BigQuery."""
    table_expression = self.ibis_client.table(table)
    return DataFrame(self, table_expression)


class DataFrame:
  """A deferred DataFrame, representing data and cloud transformations."""

  def __init__(
      self,
      engine: Engine,
      table: ibis_types.Table,
  ):
    self._engine = engine
    self._table = table

  def head(self, max_results: Optional[int] = 5) -> pandas.DataFrame:
    """Executes deferred operations and downloads a specific number of rows."""
    sql = self._table.compile()
    return self._engine.bqclient.query(sql).to_dataframe(
        max_results=max_results)

  def compute(self) -> pandas.DataFrame:
    """Executes deferred operations and downloads the results."""
    sql = self._table.compile()
    return self._engine.bqclient.query(sql).to_dataframe()


def connect(context: Optional[Context] = None) -> Engine:
  return Engine(context)
