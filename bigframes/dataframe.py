"""DataFrame is a two dimensional data structure."""

import typing
from typing import Optional

from ibis.expr.types import Table
import pandas

# Avoid circular import just to type check.
if typing.TYPE_CHECKING:
    from bigframes.engine import Engine


class DataFrame:
    """A deferred DataFrame, representing data and cloud transformations."""

    def __init__(
        self,
        engine: "Engine",
        table: Table,
    ):
        self._engine = engine
        self._table = table

    def head(self, max_results: Optional[int] = 5) -> pandas.DataFrame:
        """Executes deferred operations and downloads a specific number of rows."""
        sql = self._table.compile()
        return self._engine.bqclient.query(sql).to_dataframe(max_results=max_results)

    def compute(self) -> pandas.DataFrame:
        """Executes deferred operations and downloads the results."""
        sql = self._table.compile()
        return self._engine.bqclient.query(sql).to_dataframe()
