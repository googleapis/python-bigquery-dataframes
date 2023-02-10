import typing

import ibis.expr.types as ibis_types


class Scalar:
    """A possibly deferred scalar object."""

    def __init__(self, value: ibis_types.Scalar):
        self._value = value

    def compute(self) -> typing.Any:
        """Executes deferred operations and downloads the resulting scalar."""
        return self._value.execute()
