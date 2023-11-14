import pytest

from bigframes.core import log_adapter

MAX_LABELS_COUNT = 64


@pytest.fixture
def test_instance():
    # Create a simple class for testing
    @log_adapter.class_logger()
    class TestClass:
        def method1(self):
            pass

        def method2(self):
            pass

    return TestClass()


def test_class_logger_decorator(test_instance):
    # Ensure that the class logger decorator adds _api_methods attribute
    assert hasattr(test_instance, "_api_methods")
    assert test_instance._api_methods == []


def test_add_api_method(test_instance):
    # Ensure that add_api_method correctly adds a method to _api_methods
    log_adapter.add_api_method("method3", test_instance)
    assert test_instance._api_methods == ["method3"]


def test_add_api_method_limit(test_instance):
    # Ensure that add_api_method correctly adds a method to _api_methods
    for i in range(70):
        log_adapter.add_api_method("method3", test_instance)
    assert len(test_instance._api_methods) == MAX_LABELS_COUNT


def test_get_and_reset_api_methods(test_instance):
    # Ensure that get_and_reset_api_methods returns a copy and resets the list
    test_instance.method1()
    test_instance.method2()
    previous_methods = log_adapter.get_and_reset_api_methods(test_instance)
    assert previous_methods == ["method2", "method1"]
    assert test_instance._api_methods == []
