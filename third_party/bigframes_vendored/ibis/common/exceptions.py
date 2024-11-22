# Contains code from https://github.com/ibis-project/ibis/blob/9.2.0/ibis/common/exceptions.py

from __future__ import annotations


class IbisError(Exception):
    """IbisError."""


class InternalError(IbisError):
    """InternalError."""


class IntegrityError(IbisError):
    """IntegrityError."""


class ExpressionError(IbisError):
    """ExpressionError."""


class RelationError(ExpressionError):
    """RelationError."""


class TranslationError(IbisError):
    """TranslationError."""


class OperationNotDefinedError(TranslationError):
    """OperationNotDefinedError."""


class UnsupportedOperationError(TranslationError):
    """UnsupportedOperationError."""


class UnsupportedBackendType(TranslationError):
    """UnsupportedBackendType."""


class UnboundExpressionError(ValueError, IbisError):
    """UnboundExpressionError."""


class IbisInputError(ValueError, IbisError):
    """IbisInputError."""


class IbisTypeError(TypeError, IbisError):
    """IbisTypeError."""


class InputTypeError(IbisTypeError):
    """InputTypeError."""


class UnsupportedArgumentError(IbisError):
    """UnsupportedArgumentError."""
