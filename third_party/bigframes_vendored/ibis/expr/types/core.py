# Contains code from https://github.com/ibis-project/ibis/blob/9.2.0/ibis/expr/types/core.py

from __future__ import annotations

import contextlib
import os
from typing import Any, NoReturn, TYPE_CHECKING
import webbrowser

import bigframes_vendored.ibis
from bigframes_vendored.ibis.common.annotations import ValidationError
from bigframes_vendored.ibis.common.exceptions import IbisError, TranslationError
from bigframes_vendored.ibis.common.grounds import Immutable
from bigframes_vendored.ibis.common.patterns import Coercible, CoercionError
from bigframes_vendored.ibis.common.typing import get_defining_scope
from bigframes_vendored.ibis.config import _default_backend
from bigframes_vendored.ibis.config import options as opts
from bigframes_vendored.ibis.expr.format import pretty
import bigframes_vendored.ibis.expr.operations as ops
from bigframes_vendored.ibis.expr.types.pretty import to_rich
from bigframes_vendored.ibis.util import experimental
import pandas as pd
from public import public
from rich.console import Console
from rich.jupyter import JupyterMixin
from rich.text import Text

if TYPE_CHECKING:
    from collections.abc import Iterator, Mapping
    from pathlib import Path

    from bigframes_vendored.ibis.backends import BaseBackend
    import bigframes_vendored.ibis.expr.types as ir
    from bigframes_vendored.ibis.expr.visualize import (
        EdgeAttributeGetter,
        NodeAttributeGetter,
    )
    import polars as pl
    import pyarrow as pa
    import torch


class _FixedTextJupyterMixin(JupyterMixin):
    """JupyterMixin adds a spurious newline to text, this fixes the issue."""

    def _repr_mimebundle_(self, *args, **kwargs):
        bundle = super()._repr_mimebundle_(*args, **kwargs)
        bundle["text/plain"] = bundle["text/plain"].rstrip()
        return bundle


@public
class Expr(Immutable, Coercible):
    """Base expression class."""

    __slots__ = ("_arg",)
    _arg: ops.Node

    def _noninteractive_repr(self) -> str:
        if bigframes_vendored.ibis.options.repr.show_variables:
            scope = get_defining_scope(self, types=Expr)
        else:
            scope = None
        return pretty(self.op(), scope=scope)

    def _interactive_repr(self) -> str:
        console = Console(force_terminal=False)
        with console.capture() as capture:
            try:
                console.print(self)
            except TranslationError as e:
                lines = [
                    "Translation to backend failed",
                    f"Error message: {e!r}",
                    "Expression repr follows:",
                    self._noninteractive_repr(),
                ]
                return "\n".join(lines)
        return capture.get().rstrip()

    def __repr__(self) -> str:
        if bigframes_vendored.ibis.options.interactive:
            return self._interactive_repr()
        else:
            return self._noninteractive_repr()

    def __rich_console__(self, console: Console, options):
        if console.is_jupyter:
            # Rich infers a console width in jupyter notebooks, but since
            # notebooks can use horizontal scroll bars we don't want to apply a
            # limit here. Since rich requires an integer for max_width, we
            # choose an arbitrarily large integer bound. Note that we need to
            # handle this here rather than in `to_rich`, as this setting
            # also needs to be forwarded to `console.render`.
            options = options.update(max_width=1_000_000)
            console_width = None
        else:
            console_width = options.max_width

        try:
            if opts.interactive:
                rich_object = to_rich(self, console_width=console_width)
            else:
                rich_object = Text(self._noninteractive_repr())
        except Exception as e:
            # In IPython exceptions inside of _repr_mimebundle_ are swallowed to
            # allow calling several display functions and choosing to display
            # the "best" result based on some priority.
            # This behavior, though, means that exceptions that bubble up inside of the interactive repr
            # are silently caught.
            #
            # We can't stop the exception from being swallowed, but we can force
            # the display of that exception as we do here.
            #
            # A _very_ annoying caveat is that this exception is _not_ being
            # ` raise`d, it is only being printed to the console.  This means
            # that you cannot "catch" it.
            #
            # This restriction is only present in IPython, not in other REPLs.
            console.print_exception()
            raise e
        return console.render(rich_object, options=options)

    def __init__(self, arg: ops.Node) -> None:
        object.__setattr__(self, "_arg", arg)

    def __iter__(self) -> NoReturn:
        raise TypeError(f"{self.__class__.__name__!r} object is not iterable")

    @classmethod
    def __coerce__(cls, value):
        if isinstance(value, cls):
            return value
        elif isinstance(value, ops.Node):
            return value.to_expr()
        else:
            raise CoercionError("Unable to coerce value to an expression")

    def __reduce__(self):
        return (self.__class__, (self._arg,))

    def __hash__(self):
        return hash((self.__class__, self._arg))

    def equals(self, other):
        """Return whether this expression is _structurally_ equivalent to `other`.

        If you want to produce an equality expression, use `==` syntax.

        Parameters
        ----------
        other
            Another expression

        Examples
        --------
        >>> import ibis
        >>> t1 = ibis.table(dict(a="int"), name="t")
        >>> t2 = ibis.table(dict(a="int"), name="t")
        >>> t1.equals(t2)
        True
        >>> v = ibis.table(dict(a="string"), name="v")
        >>> t1.equals(v)
        False
        """
        if not isinstance(other, Expr):
            raise TypeError(
                f"invalid equality comparison between Expr and {type(other)}"
            )
        return self._arg.equals(other._arg)

    def __bool__(self) -> bool:
        raise ValueError("The truth value of an Ibis expression is not defined")

    __nonzero__ = __bool__

    def has_name(self):
        """Check whether this expression has an explicit name."""
        return hasattr(self._arg, "name")

    def get_name(self):
        """Return the name of this expression."""
        return self._arg.name

    def _repr_png_(self) -> bytes | None:
        if opts.interactive or not opts.graphviz_repr:
            return None
        try:
            import bigframes_vendored.ibis.expr.visualize as viz
        except ImportError:
            return None
        else:
            # Something may go wrong, and we can't error in the notebook
            # so fallback to the default text representation.
            with contextlib.suppress(Exception):
                return viz.to_graph(self).pipe(format="png")

    def visualize(
        self,
        format: str = "svg",
        *,
        label_edges: bool = False,
        verbose: bool = False,
        node_attr: Mapping[str, str] | None = None,
        node_attr_getter: NodeAttributeGetter | None = None,
        edge_attr: Mapping[str, str] | None = None,
        edge_attr_getter: EdgeAttributeGetter | None = None,
    ) -> None:
        """Visualize an expression as a GraphViz graph in the browser.

        Parameters
        ----------
        format
            Image output format. These are specified by the ``graphviz`` Python
            library.
        label_edges
            Show operation input names as edge labels
        verbose
            Print the graphviz DOT code to stderr if [](`True`)
        node_attr
            Mapping of ``(attribute, value)`` pairs set for all nodes.
            Options are specified by the ``graphviz`` Python library.
        node_attr_getter
            Callback taking a node and returning a mapping of ``(attribute, value)`` pairs
            for that node. Options are specified by the ``graphviz`` Python library.
        edge_attr
            Mapping of ``(attribute, value)`` pairs set for all edges.
            Options are specified by the ``graphviz`` Python library.
        edge_attr_getter
            Callback taking two adjacent nodes and returning a mapping of ``(attribute, value)`` pairs
            for the edge between those nodes. Options are specified by the ``graphviz`` Python library.

        Examples
        --------
        Open the visualization of an expression in default browser:

        >>> import ibis
        >>> import ibis.expr.operations as ops
        >>> left = ibis.table(dict(a="int64", b="string"), name="left")
        >>> right = ibis.table(dict(b="string", c="int64", d="string"), name="right")
        >>> expr = left.inner_join(right, "b").select(left.a, b=right.c, c=right.d)
        >>> expr.visualize(
        ...     format="svg",
        ...     label_edges=True,
        ...     node_attr={"fontname": "Roboto Mono", "fontsize": "10"},
        ...     node_attr_getter=lambda node: isinstance(node, ops.Field) and {"shape": "oval"},
        ...     edge_attr={"fontsize": "8"},
        ...     edge_attr_getter=lambda u, v: isinstance(u, ops.Field) and {"color": "red"},
        ... )  # quartodoc: +SKIP # doctest: +SKIP

        Raises
        ------
        ImportError
            If ``graphviz`` is not installed.
        """
        import bigframes_vendored.ibis.expr.visualize as viz

        path = viz.draw(
            viz.to_graph(
                self,
                node_attr=node_attr,
                node_attr_getter=node_attr_getter,
                edge_attr=edge_attr,
                edge_attr_getter=edge_attr_getter,
                label_edges=label_edges,
            ),
            format=format,
            verbose=verbose,
        )
        webbrowser.open(f"file://{os.path.abspath(path)}")

    def pipe(self, f, *args: Any, **kwargs: Any) -> Expr:
        """Compose `f` with `self`.

        Parameters
        ----------
        f
            If the expression needs to be passed as anything other than the
            first argument to the function, pass a tuple with the argument
            name. For example, (f, 'data') if the function f expects a 'data'
            keyword
        args
            Positional arguments to `f`
        kwargs
            Keyword arguments to `f`

        Examples
        --------
        >>> import ibis
        >>> t = ibis.table([("a", "int64"), ("b", "string")], name="t")
        >>> f = lambda a: (a + 1).name("a")
        >>> g = lambda a: (a * 2).name("a")
        >>> result1 = t.a.pipe(f).pipe(g)
        >>> result1
        r0 := UnboundTable: t
          a int64
          b string
        a: r0.a + 1 * 2

        >>> result2 = g(f(t.a))  # equivalent to the above
        >>> result1.equals(result2)
        True

        Returns
        -------
        Expr
            Result type of passed function
        """
        if isinstance(f, tuple):
            f, data_keyword = f
            kwargs = kwargs.copy()
            kwargs[data_keyword] = self
            return f(*args, **kwargs)
        else:
            return f(self, *args, **kwargs)

    def op(self) -> ops.Node:
        return self._arg

    def _find_backends(self) -> tuple[list[BaseBackend], bool]:
        """Return the possible backends for an expression.

        Returns
        -------
        list[BaseBackend]
            A list of the backends found.
        """

        backends = set()
        has_unbound = False
        node_types = (ops.UnboundTable, ops.DatabaseTable, ops.SQLQueryResult)
        for table in self.op().find(node_types):
            if isinstance(table, ops.UnboundTable):
                has_unbound = True
            else:
                backends.add(table.source)

        return list(backends), has_unbound

    def _find_backend(self, *, use_default: bool = False) -> BaseBackend:
        """Find the backend attached to an expression.

        Parameters
        ----------
        use_default
            If [](`True`) and the default backend isn't set, initialize the
            default backend and use that. This should only be set to `True` for
            `.execute()`. For other contexts such as compilation, this option
            doesn't make sense so the default value is [](`False`).

        Returns
        -------
        BaseBackend
            A backend that is attached to the expression
        """
        backends, has_unbound = self._find_backends()

        if not backends:
            if has_unbound:
                raise IbisError(
                    "Expression contains unbound tables and therefore cannot "
                    "be executed. Use `<backend>.execute(expr)` to execute "
                    "against an explicit backend, or rebuild the expression "
                    "using bound tables instead."
                )
            default = _default_backend() if use_default else None
            if default is None:
                raise IbisError(
                    "Expression depends on no backends, and found no default"
                )
            return default

        if len(backends) > 1:
            raise IbisError("Multiple backends found for this expression")

        return backends[0]

    def execute(
        self,
        limit: int | str | None = "default",
        params: Mapping[ir.Value, Any] | None = None,
        **kwargs: Any,
    ):
        """Execute an expression against its backend if one exists.

        Parameters
        ----------
        limit
            An integer to effect a specific row limit. A value of `None` means
            "no limit". The default is in `ibis/config.py`.
        params
            Mapping of scalar parameter expressions to value
        kwargs
            Keyword arguments
        """
        return self._find_backend(use_default=True).execute(
            self, limit=limit, params=params, **kwargs
        )

    def compile(
        self,
        limit: int | None = None,
        params: Mapping[ir.Value, Any] | None = None,
        pretty: bool = False,
    ):
        """Compile to an execution target.

        Parameters
        ----------
        limit
            An integer to effect a specific row limit. A value of `None` means
            "no limit". The default is in `ibis/config.py`.
        params
            Mapping of scalar parameter expressions to value
        pretty
            In case of SQL backends, return a pretty formatted SQL query.
        """
        return self._find_backend().compile(
            self, limit=limit, params=params, pretty=pretty
        )

    @experimental
    def to_pyarrow_batches(
        self,
        *,
        limit: int | str | None = None,
        params: Mapping[ir.Value, Any] | None = None,
        chunk_size: int = 1_000_000,
        **kwargs: Any,
    ) -> pa.ipc.RecordBatchReader:
        """Execute expression and return a RecordBatchReader.

        This method is eager and will execute the associated expression
        immediately.

        Parameters
        ----------
        limit
            An integer to effect a specific row limit. A value of `None` means
            "no limit". The default is in `ibis/config.py`.
        params
            Mapping of scalar parameter expressions to value.
        chunk_size
            Maximum number of rows in each returned record batch.
        kwargs
            Keyword arguments

        Returns
        -------
        results
            RecordBatchReader
        """
        return self._find_backend(use_default=True).to_pyarrow_batches(
            self,
            params=params,
            limit=limit,
            chunk_size=chunk_size,
            **kwargs,
        )

    @experimental
    def to_pyarrow(
        self,
        *,
        params: Mapping[ir.Scalar, Any] | None = None,
        limit: int | str | None = None,
        **kwargs: Any,
    ) -> pa.Table:
        """Execute expression and return results in as a pyarrow table.

        This method is eager and will execute the associated expression
        immediately.

        Parameters
        ----------
        params
            Mapping of scalar parameter expressions to value.
        limit
            An integer to effect a specific row limit. A value of `None` means
            "no limit". The default is in `ibis/config.py`.
        kwargs
            Keyword arguments

        Returns
        -------
        Table
            A pyarrow table holding the results of the executed expression.
        """
        return self._find_backend(use_default=True).to_pyarrow(
            self, params=params, limit=limit, **kwargs
        )

    @experimental
    def to_polars(
        self,
        *,
        params: Mapping[ir.Scalar, Any] | None = None,
        limit: int | str | None = None,
        **kwargs: Any,
    ) -> pl.DataFrame:
        """Execute expression and return results as a polars dataframe.

        This method is eager and will execute the associated expression
        immediately.

        Parameters
        ----------
        params
            Mapping of scalar parameter expressions to value.
        limit
            An integer to effect a specific row limit. A value of `None` means
            "no limit". The default is in `ibis/config.py`.
        kwargs
            Keyword arguments

        Returns
        -------
        DataFrame
            A polars dataframe holding the results of the executed expression.
        """
        return self._find_backend(use_default=True).to_polars(
            self, params=params, limit=limit, **kwargs
        )

    @experimental
    def to_pandas_batches(
        self,
        *,
        limit: int | str | None = None,
        params: Mapping[ir.Value, Any] | None = None,
        chunk_size: int = 1_000_000,
        **kwargs: Any,
    ) -> Iterator[pd.DataFrame | pd.Series | Any]:
        """Execute expression and return an iterator of pandas DataFrames.

        This method is eager and will execute the associated expression
        immediately.

        Parameters
        ----------
        limit
            An integer to effect a specific row limit. A value of `None` means
            "no limit". The default is in `ibis/config.py`.
        params
            Mapping of scalar parameter expressions to value.
        chunk_size
            Maximum number of rows in each returned `DataFrame``.
        kwargs
            Keyword arguments

        Returns
        -------
        Iterator[pd.DataFrame]
        """
        return self._find_backend(use_default=True).to_pandas_batches(
            self,
            params=params,
            limit=limit,
            chunk_size=chunk_size,
            **kwargs,
        )

    @experimental
    def to_parquet(
        self,
        path: str | Path,
        *,
        params: Mapping[ir.Scalar, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        """Write the results of executing the given expression to a parquet file.

        This method is eager and will execute the associated expression
        immediately.

        Parameters
        ----------
        path
            The data source. A string or Path to the parquet file.
        params
            Mapping of scalar parameter expressions to value.
        **kwargs
            Additional keyword arguments passed to pyarrow.parquet.ParquetWriter

        https://arrow.apache.org/docs/python/generated/pyarrow.parquet.ParquetWriter.html

        Examples
        --------
        Write out an expression to a single parquet file.

        >>> import ibis
        >>> import tempfile
        >>> penguins = ibis.examples.penguins.fetch()
        >>> penguins.to_parquet(tempfile.mktemp())

        Partition on a single column.

        >>> penguins.to_parquet(tempfile.mkdtemp(), partition_by="year")

        Partition on multiple columns.

        >>> penguins.to_parquet(tempfile.mkdtemp(), partition_by=("year", "island"))

        ::: {.callout-note}
        ## Hive-partitioned output is currently only supported when using DuckDB
        :::
        """
        self._find_backend(use_default=True).to_parquet(self, path, **kwargs)

    @experimental
    def to_csv(
        self,
        path: str | Path,
        *,
        params: Mapping[ir.Scalar, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        """Write the results of executing the given expression to a CSV file.

        This method is eager and will execute the associated expression
        immediately.

        Parameters
        ----------
        path
            The data source. A string or Path to the CSV file.
        params
            Mapping of scalar parameter expressions to value.
        **kwargs
            Additional keyword arguments passed to pyarrow.csv.CSVWriter

        https://arrow.apache.org/docs/python/generated/pyarrow.csv.CSVWriter.html
        """
        self._find_backend(use_default=True).to_csv(self, path, **kwargs)

    @experimental
    def to_delta(
        self,
        path: str | Path,
        *,
        params: Mapping[ir.Scalar, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        """Write the results of executing the given expression to a Delta Lake table.

        This method is eager and will execute the associated expression
        immediately.

        Parameters
        ----------
        path
            The data source. A string or Path to the Delta Lake table directory.
        params
            Mapping of scalar parameter expressions to value.
        **kwargs
            Additional keyword arguments passed to deltalake.writer.write_deltalake method
        """
        self._find_backend(use_default=True).to_delta(self, path, **kwargs)

    @experimental
    def to_torch(
        self,
        *,
        params: Mapping[ir.Scalar, Any] | None = None,
        limit: int | str | None = None,
        **kwargs: Any,
    ) -> dict[str, torch.Tensor]:
        """Execute an expression and return results as a dictionary of torch tensors.

        Parameters
        ----------
        params
            Parameters to substitute into the expression.
        limit
            An integer to effect a specific row limit. A value of `None` means no limit.
        kwargs
            Keyword arguments passed into the backend's `to_torch` implementation.

        Returns
        -------
        dict[str, torch.Tensor]
            A dictionary of torch tensors, keyed by column name.
        """
        return self._find_backend(use_default=True).to_torch(
            self, params=params, limit=limit, **kwargs
        )

    def unbind(self) -> ir.Table:
        """Return an expression built on `UnboundTable` instead of backend-specific objects."""
        from bigframes_vendored.ibis.expr.rewrites import _, d, p

        rule = p.DatabaseTable >> d.UnboundTable(
            name=_.name, schema=_.schema, namespace=_.namespace
        )
        return self.op().replace(rule).to_expr()

    def as_table(self) -> ir.Table:
        """Convert an expression to a table."""
        raise NotImplementedError(
            f"{type(self)} expressions cannot be converted into tables"
        )

    def as_scalar(self) -> ir.Scalar:
        """Convert an expression to a scalar."""
        raise NotImplementedError(
            f"{type(self)} expressions cannot be converted into scalars"
        )


def _binop(op_class: type[ops.Binary], left: ir.Value, right: ir.Value) -> ir.Value:
    """Try to construct a binary operation.

    Parameters
    ----------
    op_class
        The `ops.Binary` subclass for the operation
    left
        Left operand
    right
        Right operand

    Returns
    -------
    ir.Value
        A value expression

    Examples
    --------
    >>> import ibis
    >>> import ibis.expr.operations as ops
    >>> expr = _binop(ops.TimeAdd, ibis.time("01:00"), ibis.interval(hours=1))
    >>> expr
    TimeAdd(datetime.time(1, 0), 1h): datetime.time(1, 0) + 1 h
    >>> _binop(ops.TimeAdd, 1, ibis.interval(hours=1))
    TimeAdd(datetime.time(0, 0, 1), 1h): datetime.time(0, 0, 1) + 1 h
    """
    try:
        node = op_class(left, right)
    except (ValidationError, NotImplementedError):
        return NotImplemented
    else:
        return node.to_expr()


def _is_null_literal(value: Any) -> bool:
    """Detect whether `value` will be treated by ibis as a null literal."""
    if isinstance(value, Expr):
        op = value.op()
        return isinstance(op, ops.Literal) and op.value is None
    if pd.isna(value):
        return True
    return False
