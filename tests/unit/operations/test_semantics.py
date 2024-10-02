# Copyright 2024 Google LLC
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

import pandas as pd
import pandas.testing
import pytest

import bigframes
import bigframes.dataframe as dataframe
import bigframes.ml.llm as llm

_MODEL = llm.GeminiTextGenerator(model_name=llm._GEMINI_1P5_FLASH_001_ENDPOINT)


def test_semantics_experiment_off_raise_error():
    bigframes.options.experiments.semantic_operators = False
    df = dataframe.DataFrame(
        {"country": ["USA", "Germany"], "city": ["Seattle", "Berlin"]}
    )

    with pytest.raises(NotImplementedError):
        df.semantics


def test_filter():
    bigframes.options.experiments.semantic_operators = True
    df = dataframe.DataFrame(
        {"country": ["USA", "Germany"], "city": ["Seattle", "Berlin"]}
    )

    actual_df = df.semantics.filter(
        "{city} is the capital of {country}", _MODEL
    ).to_pandas()

    expected_df = pd.DataFrame({"country": ["Germany"], "city": ["Berlin"]}, index=[1])
    pandas.testing.assert_frame_equal(
        actual_df, expected_df, check_dtype=False, check_index_type=False
    )
