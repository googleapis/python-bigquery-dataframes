from __future__ import annotations

from bigframes_vendored.sqlglot import exp
from bigframes_vendored.sqlglot.typing import EXPRESSION_METADATA

EXPRESSION_METADATA = {
    **EXPRESSION_METADATA,
    exp.If: {
        "annotator": lambda self, e: self._annotate_by_args(
            e, "true", "false", promote=True
        )
    },
    exp.Coalesce: {
        "annotator": lambda self, e: self._annotate_by_args(
            e, "this", "expressions", promote=True
        )
    },
}
