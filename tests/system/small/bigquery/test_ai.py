# Copyright 2025 Google LLC
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

from packaging import version
import pandas as pd
import pyarrow as pa
import pytest
import sqlglot

from bigframes import dtypes, series
import bigframes.bigquery as bbq
import bigframes.pandas as bpd


def test_ai_function_pandas_input(session):
    s1 = pd.Series(["apple", "bear"])
    s2 = bpd.Series(["fruit", "tree"], session=session)
    prompt = (s1, " is a ", s2)

    result = bbq.ai.generate_bool(prompt, endpoint="gemini-2.5-flash")

    assert _contains_no_nulls(result)
    assert result.dtype == pd.ArrowDtype(
        pa.struct(
            (
                pa.field("result", pa.bool_()),
                pa.field("full_response", dtypes.JSON_ARROW_TYPE),
                pa.field("status", pa.string()),
            )
        )
    )


def test_ai_function_compile_model_params(session):
    if version.Version(sqlglot.__version__) < version.Version("25.18.0"):
        pytest.skip(
            "Skip test because SQLGLot cannot compile model params to JSON at this version."
        )

    s1 = bpd.Series(["apple", "bear"], session=session)
    s2 = bpd.Series(["fruit", "tree"], session=session)
    prompt = (s1, " is a ", s2)
    model_params = {"generation_config": {"thinking_config": {"thinking_budget": 0}}}

    result = bbq.ai.generate_bool(
        prompt, endpoint="gemini-2.5-flash", model_params=model_params
    )

    assert _contains_no_nulls(result)
    assert result.dtype == pd.ArrowDtype(
        pa.struct(
            (
                pa.field("result", pa.bool_()),
                pa.field("full_response", dtypes.JSON_ARROW_TYPE),
                pa.field("status", pa.string()),
            )
        )
    )


def test_ai_generate_bool(session):
    s1 = bpd.Series(["apple", "bear"], session=session)
    s2 = bpd.Series(["fruit", "tree"], session=session)
    prompt = (s1, " is a ", s2)

    result = bbq.ai.generate_bool(prompt, endpoint="gemini-2.5-flash")

    assert _contains_no_nulls(result)
    assert result.dtype == pd.ArrowDtype(
        pa.struct(
            (
                pa.field("result", pa.bool_()),
                pa.field("full_response", dtypes.JSON_ARROW_TYPE),
                pa.field("status", pa.string()),
            )
        )
    )


def test_ai_generate_bool_multi_model(session):
    df = session.from_glob_path(
        "gs://bigframes-dev-testing/a_multimodel/images/*", name="image"
    )

    result = bbq.ai.generate_bool((df["image"], " contains an animal"))

    assert _contains_no_nulls(result)
    assert result.dtype == pd.ArrowDtype(
        pa.struct(
            (
                pa.field("result", pa.bool_()),
                pa.field("full_response", dtypes.JSON_ARROW_TYPE),
                pa.field("status", pa.string()),
            )
        )
    )


def test_ai_generate_int(session):
    s = bpd.Series(["Cat"], session=session)
    prompt = ("How many legs does a ", s, " have?")

    result = bbq.ai.generate_int(prompt, endpoint="gemini-2.5-flash")

    assert _contains_no_nulls(result)
    assert result.dtype == pd.ArrowDtype(
        pa.struct(
            (
                pa.field("result", pa.int64()),
                pa.field("full_response", dtypes.JSON_ARROW_TYPE),
                pa.field("status", pa.string()),
            )
        )
    )


def test_ai_generate_int_multi_model(session):
    df = session.from_glob_path(
        "gs://bigframes-dev-testing/a_multimodel/images/*", name="image"
    )

    result = bbq.ai.generate_int(
        ("How many animals are there in the picture ", df["image"])
    )

    assert _contains_no_nulls(result)
    assert result.dtype == pd.ArrowDtype(
        pa.struct(
            (
                pa.field("result", pa.int64()),
                pa.field("full_response", dtypes.JSON_ARROW_TYPE),
                pa.field("status", pa.string()),
            )
        )
    )


def _contains_no_nulls(s: series.Series) -> bool:
    return len(s) == s.count()
