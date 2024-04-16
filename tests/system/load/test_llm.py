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

import bigframes.ml.llm


def test_llm_palm_configure_fit(llm_fine_tune_df_default_index, dataset_id):
    model = bigframes.ml.llm.PaLM2TextGenerator(
        model_name="text-bison", max_iterations=1, evaluation_task="CLASSIFICATION"
    )

    df = llm_fine_tune_df_default_index.dropna()
    X_train = df[["prompt"]]
    y_train = df[["label"]]
    model.fit(X_train, y_train)

    # save, load, check parameters to ensure configuration was kept
    reloaded_model = model.to_gbq(
        f"{dataset_id}.temp_configured_palm_model", replace=True
    )
    assert (
        f"{dataset_id}.temp_configured_palm_model"
        in reloaded_model._bqml_model.model_name
    )
    assert reloaded_model.evaluation_task == "CLASSIFICATION"
