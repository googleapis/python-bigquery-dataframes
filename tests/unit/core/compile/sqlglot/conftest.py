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

import pathlib

import pandas as pd
import pytest

import tests.system.utils

CURRENT_DIR = pathlib.Path(__file__).parent
DATA_DIR = CURRENT_DIR.parent.parent.parent.parent / "data"


@pytest.fixture(scope="session")
def compiler_session():
    from . import compiler_session

    return compiler_session.SQLCompilerSession()


@pytest.fixture(scope="session")
def scalars_types_pandas_df() -> pd.DataFrame:
    """pd.DataFrame with all scalar types and rowindex as index."""
    # TODO: all types pandas dataframes
    # TODO: add tests for empty dataframes
    df = pd.read_json(
        DATA_DIR / "scalars.jsonl",
        lines=True,
    )
    tests.system.utils.convert_pandas_dtypes(df, bytes_col=True)

    # add more complexity index.
    df = df.set_index("rowindex", drop=False)
    df.index.name = None
    return df
