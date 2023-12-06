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
import pytest

from bigframes.apps.first_party.synthetic_data_generator.core.prompt_generator import (
    PromptGenerator,
)


def test_initialization_default():
    pg = PromptGenerator()
    assert pg.get_num_rows() == 100
    assert pg.get_columns() == []


def test_initialization_with_dataframe_info():
    dataframe_info = {
        "num_rows": 500,
        "columns": [("col1", "int", "description1"), ("col2", "str", "description2")],
    }
    pg = PromptGenerator(dataframe_info)
    assert pg.get_num_rows() == 500
    assert pg.get_columns() == dataframe_info["columns"]


def test_invalid_dataframe_info():
    with pytest.raises(ValueError):
        PromptGenerator({"num_rows": "not an int", "columns": []})


def test_add_delete_column():
    pg = PromptGenerator()
    pg.add_column("col1", "type1", "description1")
    assert pg.get_columns() == [("col1", "type1", "description1")]
    pg.delete_column("col1")
    assert pg.get_columns() == []


def test_update_num_rows():
    pg = PromptGenerator()
    pg.update_num_rows(200)
    assert pg.get_num_rows() == 200


def test_generate_prompt():
    dataframe_info = {"num_rows": 100, "columns": [("col1", "int", "description1")]}
    prompt = PromptGenerator(dataframe_info).generate_prompt()
    assert "Num rows: 100" in prompt
    assert "Num columns: 1" in prompt
    assert "column name: col1, type: int, description: description1" in prompt
