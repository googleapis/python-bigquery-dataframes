# Contains code from https://github.com/ibis-project/ibis/blob/9.2.0/ibis/expr/operations/core.py

from __future__ import annotations

from abc import abstractmethod
from typing import Generic, Optional

from bigframes_vendored.ibis.common.annotations import attribute
from bigframes_vendored.ibis.common.graph import Node as Traversable
from bigframes_vendored.ibis.common.grounds import Concrete
from bigframes_vendored.ibis.common.patterns import Coercible, CoercionError
from bigframes_vendored.ibis.common.typing import DefaultTypeVars
import bigframes_vendored.ibis.expr.datashape as ds
import bigframes_vendored.ibis.expr.datatypes as dt
import bigframes_vendored.ibis.expr.rules as rlz
from bigframes_vendored.ibis.util import is_iterable
from public import public
from typing_extensions import Any, Self, TypeVar


@public
class Node(Concrete, Traversable):
    def equals(self, other) -> bool:
        if not isinstance(other, Node):
            raise TypeError(
                f"invalid equality comparison between Node and {type(other)}"
            )
        return self == other

    # Avoid custom repr for performance reasons
    __repr__ = object.__repr__

    # TODO(kszucs): hidrate the __children__ traversable attribute
    # @attribute
    # def __children__(self):
    #     return super().__children__


T = TypeVar("T", bound=dt.DataType, covariant=True)
S = TypeVar("S", bound=ds.DataShape, default=ds.Any, covariant=True)


@public
class Value(Node, Coercible, DefaultTypeVars, Generic[T, S]):
    @classmethod
    def __coerce__(
        cls, value: Any, T: Optional[type] = None, S: Optional[type] = None
    ) -> Self:
        # note that S=Shape is unused here since the pattern will check the
        # shape of the value expression after executing Value.__coerce__()
        from bigframes_vendored.ibis.expr.operations.generic import Literal, NULL
        from bigframes_vendored.ibis.expr.types import Expr

        if isinstance(value, Expr):
            value = value.op()

        if isinstance(value, Value):
            if value == NULL:
                # treat the NULL literal the same as None to implicitly cast to
                # the requested datatype if any
                value = None
            else:
                return value

        if T is dt.Integer:
            dtype = dt.infer(int(value))
        elif T is dt.Floating:
            dtype = dt.infer(float(value))
        else:
            try:
                dtype = dt.DataType.from_typehint(T)
            except TypeError:
                dtype = dt.infer(value)

        try:
            return Literal(value, dtype=dtype)
        except TypeError:
            raise CoercionError(f"Unable to coerce {value!r} to Value[{T!r}]")

    # TODO(kszucs): cover it with tests
    # TODO(kszucs): figure out how to represent not named arguments
    @property
    def name(self) -> str:
        names = []
        for arg in self.__args__:
            if is_iterable(arg):
                elements = [
                    element_name
                    for element in arg
                    if (element_name := getattr(element, "name", None)) is not None
                ]
                joined = ", ".join(elements)
                fmt = "({})" if len(elements) != 1 else "({},)"
                names.append(fmt.format(joined))
            elif (name := getattr(arg, "name", None)) is not None:
                names.append(name)
        return f"{self.__class__.__name__}({', '.join(names)})"

    @property
    @abstractmethod
    def dtype(self) -> T:
        """Ibis datatype of the produced value expression.

        Returns
        -------
        dt.DataType

        """

    @property
    @abstractmethod
    def shape(self) -> S:
        """Shape of the produced value expression.

        Possible values are: "scalar" and "columnar"

        Returns
        -------
        ds.Shape

        """

    @attribute
    def relations(self):
        """Set of relations the value node depends on."""
        children = (n.relations for n in self.__children__ if isinstance(n, Value))
        return frozenset().union(*children)

    def to_expr(self):
        import bigframes_vendored.ibis.expr.types as ir

        if self.shape.is_columnar():
            typename = self.dtype.column
        else:
            typename = self.dtype.scalar

        return getattr(ir, typename)(self)


# convenience aliases
Scalar = Value[T, ds.Scalar]
Column = Value[T, ds.Columnar]


@public
class Alias(Value):
    arg: Value
    name: str

    shape = rlz.shape_like("arg")
    dtype = rlz.dtype_like("arg")


@public
class Unary(Value):
    """A unary operation."""

    arg: Value

    @attribute
    def shape(self) -> ds.DataShape:
        return self.arg.shape

    @attribute
    def relations(self):
        return self.arg.relations


@public
class Binary(Value):
    """A binary operation."""

    left: Value
    right: Value

    @attribute
    def shape(self) -> ds.DataShape:
        return max(self.left.shape, self.right.shape)

    @attribute
    def relations(self):
        return self.left.relations | self.right.relations


@public
class Argument(Value):
    name: str
    shape: ds.DataShape
    dtype: dt.DataType

    @attribute
    def param(self) -> str:
        return f"__ibis_param_{self.name}__"


public(ValueOp=Value, UnaryOp=Unary, BinaryOp=Binary, Scalar=Scalar, Column=Column)
