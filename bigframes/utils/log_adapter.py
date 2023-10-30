import functools

from loguru import logger

_log_file_path = None
_logger = logger
_api_methods = []


def class_logger(decorated_cls):
    """Decorator that adds logging functionality to each method of the class."""
    for attr_name, attr_value in decorated_cls.__dict__.items():
        if callable(attr_value):
            setattr(decorated_cls, attr_name, method_logger(attr_value))
    return decorated_cls


def method_logger(method):
    """Decorator that adds logging functionality to a method."""

    @functools.wraps(method)
    def wrapper(*args, **kwargs):
        api_method_name = str(method.__name__)
        if not api_method_name.startswith("_"):
            add_api_method(api_method_name)
        try:
            result = method(*args, **kwargs)
            return result
        except Exception as e:
            raise e

    return wrapper


def add_api_method(method: str):
    global _api_methods
    _api_methods.append(method)
