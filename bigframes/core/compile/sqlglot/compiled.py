# from __future__ import annotations

# import functools
# import itertools
# import typing
# from typing import Collection, Optional, Sequence

# import pandas
# import sqlglot.expressions as sge

# import bigframes.core.compile.aggregate_compiler as agg_compiler
# 	@@ -36,38 +32,28 @@
# import bigframes.core.compile.scalar_op_compiler as op_compilers
# import bigframes.core.expression as ex
# import bigframes.core.guid
# from bigframes.core.ordering import OrderingExpression
# import bigframes.core.sql
# from bigframes.core.window_spec import RangeWindowBounds, RowsWindowBounds, WindowSpec
# import bigframes.dtypes
# import bigframes.operations.aggregations as agg_ops

# PREDICATE_COLUMN = "bigframes_predicate"


# op_compiler = op_compilers.scalar_op_compiler


# # SQLGlot IR Builder
# class SQLGlotIR:

#     # Concret API
# 	def literal(self, value, dtype) -> sge.Expression:
# 		if value
