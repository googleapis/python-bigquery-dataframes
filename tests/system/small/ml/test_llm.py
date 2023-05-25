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


def test_create_model_and_predict_success(session, ml_connection, llm_text_df):
    # Model creation doesn't return error
    model = llm.PaLMTextGenerator(session=session, connection_name=ml_connection)
    assert model is not None

    df = model.predict(llm_text_df).compute()
    TestCase().assertSequenceEqual(df.shape, (3, 1))
    assert llm._TEXT_GENERATE_RESULT_COLUMN in df.columns
    assert any(df[llm._TEXT_GENERATE_RESULT_COLUMN].str.contains("predictions"))
    # TODO(garrettwu): test content when throttling issues (returns empty prediction for some prompts) is resolved.
