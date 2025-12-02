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

import bigframes.core.sql.ml


def test_create_model_basic(snapshot):
    sql = bigframes.core.sql.ml.create_model_ddl(
        model_name="my_project.my_dataset.my_model",
        options={"model_type": "LINEAR_REG", "input_label_cols": ["label"]},
        training_data="SELECT * FROM my_table",
    )
    snapshot.assert_match(sql, "create_model_basic.sql")


def test_create_model_replace(snapshot):
    sql = bigframes.core.sql.ml.create_model_ddl(
        model_name="my_model",
        replace=True,
        options={"model_type": "LOGISTIC_REG"},
        training_data="SELECT * FROM t",
    )
    snapshot.assert_match(sql, "create_model_replace.sql")


def test_create_model_if_not_exists(snapshot):
    sql = bigframes.core.sql.ml.create_model_ddl(
        model_name="my_model",
        if_not_exists=True,
        options={"model_type": "KMEANS"},
        training_data="SELECT * FROM t",
    )
    snapshot.assert_match(sql, "create_model_if_not_exists.sql")


def test_create_model_transform(snapshot):
    sql = bigframes.core.sql.ml.create_model_ddl(
        model_name="my_model",
        transform=["ML.STANDARD_SCALER(c1) OVER() AS c1_scaled", "c2"],
        options={"model_type": "LINEAR_REG"},
        training_data="SELECT c1, c2, label FROM t",
    )
    snapshot.assert_match(sql, "create_model_transform.sql")


def test_create_model_remote(snapshot):
    sql = bigframes.core.sql.ml.create_model_ddl(
        model_name="my_remote_model",
        connection_name="my_project.us.my_connection",
        options={"endpoint": "gemini-pro"},
        input_schema={"prompt": "STRING"},
        output_schema={"content": "STRING"},
    )
    snapshot.assert_match(sql, "create_model_remote.sql")


def test_create_model_remote_default(snapshot):
    sql = bigframes.core.sql.ml.create_model_ddl(
        model_name="my_remote_model",
        connection_name="DEFAULT",
        options={"endpoint": "gemini-pro"},
    )
    snapshot.assert_match(sql, "create_model_remote_default.sql")


def test_create_model_training_data_and_holiday(snapshot):
    sql = bigframes.core.sql.ml.create_model_ddl(
        model_name="my_arima_model",
        options={"model_type": "ARIMA_PLUS"},
        training_data="SELECT * FROM sales",
        custom_holiday="SELECT * FROM holidays",
    )
    snapshot.assert_match(sql, "create_model_training_data_and_holiday.sql")


def test_create_model_list_option(snapshot):
    sql = bigframes.core.sql.ml.create_model_ddl(
        model_name="my_model",
        options={"hidden_units": [32, 16], "dropout": 0.2},
        training_data="SELECT * FROM t",
    )
    snapshot.assert_match(sql, "create_model_list_option.sql")
