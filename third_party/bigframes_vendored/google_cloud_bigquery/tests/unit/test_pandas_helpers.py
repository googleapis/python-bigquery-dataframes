# Original: https://github.com/googleapis/python-bigquery/blob/main/tests/unit/test__pandas_helpers.py
# Copyright 2019 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import functools
import warnings

from google.cloud.bigquery import schema
import pyarrow
import pyarrow.parquet
import pyarrow.types
import pytest


@pytest.fixture
def module_under_test():
    from third_party.bigframes_vendored.google_cloud_bigquery import _pandas_helpers

    return _pandas_helpers


def is_none(value):
    return value is None


def is_datetime(type_):
    # See: https://cloud.google.com/bigquery/docs/reference/standard-sql/data-types#datetime-type
    return all_(
        pyarrow.types.is_timestamp,
        lambda type_: type_.unit == "us",
        lambda type_: type_.tz is None,
    )(type_)


def is_numeric(type_):
    # See: https://cloud.google.com/bigquery/docs/reference/standard-sql/data-types#numeric-type
    return all_(
        pyarrow.types.is_decimal,
        lambda type_: type_.precision == 38,
        lambda type_: type_.scale == 9,
    )(type_)


def is_bignumeric(type_):
    # See: https://cloud.google.com/bigquery/docs/reference/standard-sql/data-types#numeric-type
    return all_(
        pyarrow.types.is_decimal,
        lambda type_: type_.precision == 76,
        lambda type_: type_.scale == 38,
    )(type_)


def is_timestamp(type_):
    # See: https://cloud.google.com/bigquery/docs/reference/standard-sql/data-types#timestamp-type
    return all_(
        pyarrow.types.is_timestamp,
        lambda type_: type_.unit == "us",
        lambda type_: type_.tz == "UTC",
    )(type_)


def do_all(functions, value):
    return all((func(value) for func in functions))


def all_(*functions):
    return functools.partial(do_all, functions)


def test_is_datetime():
    assert is_datetime(pyarrow.timestamp("us", tz=None))
    assert not is_datetime(pyarrow.timestamp("ms", tz=None))
    assert not is_datetime(pyarrow.timestamp("us", tz="UTC"))
    assert not is_datetime(pyarrow.timestamp("ns", tz="UTC"))
    assert not is_datetime(pyarrow.string())


def test_do_all():
    assert do_all((lambda _: True, lambda _: True), None)
    assert not do_all((lambda _: True, lambda _: False), None)
    assert not do_all((lambda _: False,), None)


def test_all_():
    assert all_(lambda _: True, lambda _: True)(None)
    assert not all_(lambda _: True, lambda _: False)(None)


@pytest.mark.parametrize(
    "bq_type,bq_mode,is_correct_type",
    [
        ("STRING", "NULLABLE", pyarrow.types.is_string),
        ("STRING", None, pyarrow.types.is_string),
        ("string", "NULLABLE", pyarrow.types.is_string),
        ("StRiNg", "NULLABLE", pyarrow.types.is_string),
        ("BYTES", "NULLABLE", pyarrow.types.is_binary),
        ("INTEGER", "NULLABLE", pyarrow.types.is_int64),
        ("INT64", "NULLABLE", pyarrow.types.is_int64),
        ("FLOAT", "NULLABLE", pyarrow.types.is_float64),
        ("FLOAT64", "NULLABLE", pyarrow.types.is_float64),
        ("NUMERIC", "NULLABLE", is_numeric),
        pytest.param(
            "BIGNUMERIC",
            "NULLABLE",
            is_bignumeric,
        ),
        ("BOOLEAN", "NULLABLE", pyarrow.types.is_boolean),
        ("BOOL", "NULLABLE", pyarrow.types.is_boolean),
        ("TIMESTAMP", "NULLABLE", is_timestamp),
        ("DATE", "NULLABLE", pyarrow.types.is_date32),
        ("TIME", "NULLABLE", pyarrow.types.is_time64),
        ("DATETIME", "NULLABLE", is_datetime),
        ("GEOGRAPHY", "NULLABLE", pyarrow.types.is_string),
        ("UNKNOWN_TYPE", "NULLABLE", is_none),
        # Use pyarrow.list_(item_type) for repeated (array) fields.
        (
            "STRING",
            "REPEATED",
            all_(
                pyarrow.types.is_list,
                lambda type_: pyarrow.types.is_string(type_.value_type),
            ),
        ),
        (
            "STRING",
            "repeated",
            all_(
                pyarrow.types.is_list,
                lambda type_: pyarrow.types.is_string(type_.value_type),
            ),
        ),
        (
            "STRING",
            "RePeAtEd",
            all_(
                pyarrow.types.is_list,
                lambda type_: pyarrow.types.is_string(type_.value_type),
            ),
        ),
        (
            "BYTES",
            "REPEATED",
            all_(
                pyarrow.types.is_list,
                lambda type_: pyarrow.types.is_binary(type_.value_type),
            ),
        ),
        (
            "INTEGER",
            "REPEATED",
            all_(
                pyarrow.types.is_list,
                lambda type_: pyarrow.types.is_int64(type_.value_type),
            ),
        ),
        (
            "INT64",
            "REPEATED",
            all_(
                pyarrow.types.is_list,
                lambda type_: pyarrow.types.is_int64(type_.value_type),
            ),
        ),
        (
            "FLOAT",
            "REPEATED",
            all_(
                pyarrow.types.is_list,
                lambda type_: pyarrow.types.is_float64(type_.value_type),
            ),
        ),
        (
            "FLOAT64",
            "REPEATED",
            all_(
                pyarrow.types.is_list,
                lambda type_: pyarrow.types.is_float64(type_.value_type),
            ),
        ),
        (
            "NUMERIC",
            "REPEATED",
            all_(pyarrow.types.is_list, lambda type_: is_numeric(type_.value_type)),
        ),
        pytest.param(
            "BIGNUMERIC",
            "REPEATED",
            all_(pyarrow.types.is_list, lambda type_: is_bignumeric(type_.value_type)),
        ),
        (
            "BOOLEAN",
            "REPEATED",
            all_(
                pyarrow.types.is_list,
                lambda type_: pyarrow.types.is_boolean(type_.value_type),
            ),
        ),
        (
            "BOOL",
            "REPEATED",
            all_(
                pyarrow.types.is_list,
                lambda type_: pyarrow.types.is_boolean(type_.value_type),
            ),
        ),
        (
            "TIMESTAMP",
            "REPEATED",
            all_(pyarrow.types.is_list, lambda type_: is_timestamp(type_.value_type)),
        ),
        (
            "DATE",
            "REPEATED",
            all_(
                pyarrow.types.is_list,
                lambda type_: pyarrow.types.is_date32(type_.value_type),
            ),
        ),
        (
            "TIME",
            "REPEATED",
            all_(
                pyarrow.types.is_list,
                lambda type_: pyarrow.types.is_time64(type_.value_type),
            ),
        ),
        (
            "DATETIME",
            "REPEATED",
            all_(pyarrow.types.is_list, lambda type_: is_datetime(type_.value_type)),
        ),
        (
            "GEOGRAPHY",
            "REPEATED",
            all_(
                pyarrow.types.is_list,
                lambda type_: pyarrow.types.is_string(type_.value_type),
            ),
        ),
        ("RECORD", "REPEATED", is_none),
        ("UNKNOWN_TYPE", "REPEATED", is_none),
    ],
)
def test_bq_to_arrow_data_type(module_under_test, bq_type, bq_mode, is_correct_type):
    field = schema.SchemaField("ignored_name", bq_type, mode=bq_mode)
    actual = module_under_test.bq_to_arrow_data_type(field)
    assert is_correct_type(actual)


@pytest.mark.parametrize("bq_type", ["RECORD", "record", "STRUCT", "struct"])
def test_bq_to_arrow_data_type_w_struct(module_under_test, bq_type):
    fields = (
        schema.SchemaField("field01", "STRING"),
        schema.SchemaField("field02", "BYTES"),
        schema.SchemaField("field03", "INTEGER"),
        schema.SchemaField("field04", "INT64"),
        schema.SchemaField("field05", "FLOAT"),
        schema.SchemaField("field06", "FLOAT64"),
        schema.SchemaField("field07", "NUMERIC"),
        schema.SchemaField("field08", "BIGNUMERIC"),
        schema.SchemaField("field09", "BOOLEAN"),
        schema.SchemaField("field10", "BOOL"),
        schema.SchemaField("field11", "TIMESTAMP"),
        schema.SchemaField("field12", "DATE"),
        schema.SchemaField("field13", "TIME"),
        schema.SchemaField("field14", "DATETIME"),
        schema.SchemaField("field15", "GEOGRAPHY"),
    )

    field = schema.SchemaField("ignored_name", bq_type, mode="NULLABLE", fields=fields)
    actual = module_under_test.bq_to_arrow_data_type(field)

    expected = (
        pyarrow.field("field01", pyarrow.string()),
        pyarrow.field("field02", pyarrow.binary()),
        pyarrow.field("field03", pyarrow.int64()),
        pyarrow.field("field04", pyarrow.int64()),
        pyarrow.field("field05", pyarrow.float64()),
        pyarrow.field("field06", pyarrow.float64()),
        pyarrow.field("field07", module_under_test.pyarrow_numeric()),
        pyarrow.field("field08", module_under_test.pyarrow_bignumeric()),
        pyarrow.field("field09", pyarrow.bool_()),
        pyarrow.field("field10", pyarrow.bool_()),
        pyarrow.field("field11", module_under_test.pyarrow_timestamp()),
        pyarrow.field("field12", pyarrow.date32()),
        pyarrow.field("field13", module_under_test.pyarrow_time()),
        pyarrow.field("field14", module_under_test.pyarrow_datetime()),
        pyarrow.field("field15", pyarrow.string()),
    )
    expected = pyarrow.struct(expected)

    assert pyarrow.types.is_struct(actual)
    assert actual.num_fields == len(fields)
    assert actual.equals(expected)


@pytest.mark.parametrize("bq_type", ["RECORD", "record", "STRUCT", "struct"])
def test_bq_to_arrow_data_type_w_array_struct(module_under_test, bq_type):
    fields = (
        schema.SchemaField("field01", "STRING"),
        schema.SchemaField("field02", "BYTES"),
        schema.SchemaField("field03", "INTEGER"),
        schema.SchemaField("field04", "INT64"),
        schema.SchemaField("field05", "FLOAT"),
        schema.SchemaField("field06", "FLOAT64"),
        schema.SchemaField("field07", "NUMERIC"),
        schema.SchemaField("field08", "BIGNUMERIC"),
        schema.SchemaField("field09", "BOOLEAN"),
        schema.SchemaField("field10", "BOOL"),
        schema.SchemaField("field11", "TIMESTAMP"),
        schema.SchemaField("field12", "DATE"),
        schema.SchemaField("field13", "TIME"),
        schema.SchemaField("field14", "DATETIME"),
        schema.SchemaField("field15", "GEOGRAPHY"),
    )

    field = schema.SchemaField("ignored_name", bq_type, mode="REPEATED", fields=fields)
    actual = module_under_test.bq_to_arrow_data_type(field)

    expected = (
        pyarrow.field("field01", pyarrow.string()),
        pyarrow.field("field02", pyarrow.binary()),
        pyarrow.field("field03", pyarrow.int64()),
        pyarrow.field("field04", pyarrow.int64()),
        pyarrow.field("field05", pyarrow.float64()),
        pyarrow.field("field06", pyarrow.float64()),
        pyarrow.field("field07", module_under_test.pyarrow_numeric()),
        pyarrow.field("field08", module_under_test.pyarrow_bignumeric()),
        pyarrow.field("field09", pyarrow.bool_()),
        pyarrow.field("field10", pyarrow.bool_()),
        pyarrow.field("field11", module_under_test.pyarrow_timestamp()),
        pyarrow.field("field12", pyarrow.date32()),
        pyarrow.field("field13", module_under_test.pyarrow_time()),
        pyarrow.field("field14", module_under_test.pyarrow_datetime()),
        pyarrow.field("field15", pyarrow.string()),
    )
    expected_value_type = pyarrow.struct(expected)

    assert pyarrow.types.is_list(actual)
    assert pyarrow.types.is_struct(actual.value_type)
    assert actual.value_type.num_fields == len(fields)
    assert actual.value_type.equals(expected_value_type)


def test_bq_to_arrow_data_type_w_struct_unknown_subfield(module_under_test):
    fields = (
        schema.SchemaField("field1", "STRING"),
        schema.SchemaField("field2", "INTEGER"),
        # Don't know what to convert UNKNOWN_TYPE to, let type inference work,
        # instead.
        schema.SchemaField("field3", "UNKNOWN_TYPE"),
    )
    field = schema.SchemaField("ignored_name", "RECORD", mode="NULLABLE", fields=fields)

    with warnings.catch_warnings(record=True) as warned:
        actual = module_under_test.bq_to_arrow_data_type(field)

    assert actual is None
    assert len(warned) == 1
    warning = warned[0]
    assert "field3" in str(warning)


def test_bq_to_arrow_field_type_override(module_under_test):
    # When loading pandas data, we may need to override the type
    # decision based on data contents, because GEOGRAPHY data can be
    # stored as either text or binary.

    assert (
        module_under_test.bq_to_arrow_field(schema.SchemaField("g", "GEOGRAPHY")).type
        == pyarrow.string()
    )

    assert (
        module_under_test.bq_to_arrow_field(
            schema.SchemaField("g", "GEOGRAPHY"),
            pyarrow.binary(),
        ).type
        == pyarrow.binary()
    )


@pytest.mark.parametrize(
    "field_type, metadata",
    [
        ("datetime", {b"ARROW:extension:name": b"google:sqlType:datetime"}),
        (
            "geography",
            {
                b"ARROW:extension:name": b"google:sqlType:geography",
                b"ARROW:extension:metadata": b'{"encoding": "WKT"}',
            },
        ),
    ],
)
def test_bq_to_arrow_field_metadata(module_under_test, field_type, metadata):
    assert (
        module_under_test.bq_to_arrow_field(
            schema.SchemaField("g", field_type)
        ).metadata
        == metadata
    )
