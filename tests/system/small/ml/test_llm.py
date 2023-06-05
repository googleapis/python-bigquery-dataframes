# Copyright 2023 Google LLC
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

from unittest import TestCase

from bigframes.ml import llm


def test_create_model_and_predict_default_params_success(
    session, ml_connection, llm_text_df
):
    # Model creation doesn't return error
    model = llm.PaLM2TextGenerator(session=session, connection_name=ml_connection)
    assert model is not None

    df = model.predict(llm_text_df).compute()
    TestCase().assertSequenceEqual(df.shape, (3, 1))
    assert "ml_generate_text_result" in df.columns
    series = df["ml_generate_text_result"]
    assert all(series.str.contains("predictions"))


def test_create_model_and_predict_with_params_success(
    session, ml_connection, llm_text_df
):
    # Model creation doesn't return error
    model = llm.PaLM2TextGenerator(session=session, connection_name=ml_connection)
    assert model is not None

    df = model.predict(
        llm_text_df, temperature=0.5, max_output_tokens=100, top_k=20, top_p=0.5
    ).compute()
    TestCase().assertSequenceEqual(df.shape, (3, 1))
    assert "ml_generate_text_result" in df.columns
    series = df["ml_generate_text_result"]
    assert all(series.str.contains("predictions"))
