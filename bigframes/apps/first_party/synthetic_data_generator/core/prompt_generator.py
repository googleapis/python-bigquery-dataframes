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

from typing import Any, Dict, List, Optional, Tuple

PROMPT_TEMPLATE = """
Write python code to generate a pandas dataframe based on the requirements:
  Num rows: {num_rows}
  Num columns: {num_columns}
{columns_info}

Note:
  - Return the code only.
  - After all import statements, declare the variable with 'num_rows = {num_rows}'.
  - Use the 'num_rows' variable to determine the number of rows in the dataframe.
  - The final dataframe should be named 'result_df'.
  - Try to use the faker library for non-numeric columns.
  - Please respect the column name semantics and the type.
  - Do not print.
"""


class PromptGenerator:
    def __init__(self, dataframe_info: Optional[Dict[str, Any]] = None) -> None:
        """
        Initialize the PromptGenerator with optional dataframe information.

        Args:
            dataframe_info (Optional[Dict[str, Any]]): A dictionary containing information
                about the dataframe, including number of rows and column details.
                Expected keys are 'num_rows' and 'columns'.
        """
        self._base_prompt = ""

        if dataframe_info:
            self._validate_dataframe_info(dataframe_info)
            self._num_rows = dataframe_info["num_rows"]
            self._columns = dataframe_info["columns"]
        else:
            self._num_rows = 100
            self._columns = []

    def _validate_dataframe_info(
        self, dataframe_info: Optional[Dict[str, Any]]
    ) -> None:
        if dataframe_info is not None:
            if "num_rows" in dataframe_info and not isinstance(
                dataframe_info["num_rows"], int
            ):
                raise ValueError("Dataframe info 'num_rows' must be an integer.")
            if "columns" in dataframe_info:
                if not isinstance(dataframe_info["columns"], list):
                    raise ValueError("Dataframe info 'columns' must be a list.")
                for column in dataframe_info["columns"]:
                    if not (
                        isinstance(column, tuple)
                        and len(column) == 3
                        and all(isinstance(item, str) for item in column)
                    ):
                        raise ValueError(
                            "Each item in 'columns' must be a tuple of three strings: (column name, dtype, description)."
                        )

    def add_column(
        self, column_name: str, column_type: str, column_description: str
    ) -> None:
        """
        Add a new column to the columns list.

        Args:
            column_name (str): The name of the column to add.
            column_description (str): The description of the column.
        """
        self._columns.append((column_name, column_type, column_description))

    def delete_column(self, column_name: str) -> None:
        """
        Delete a column from the columns list.

        Args:
            column_name (str): The name of the column to delete.
        """
        self._columns = [col for col in self._columns if col[0] != column_name]

    def update_num_rows(self, num_rows: int) -> None:
        """
        Update the number of rows.

        Args:
            num_rows (int): The new number of rows.
        """
        self._num_rows = num_rows

    def get_num_rows(self) -> int:
        """
        Get the current number of rows.

        Returns:
            int: The current number of rows.
        """
        return self._num_rows

    def get_columns(self) -> List[Tuple[str, str]]:
        """
        Get the current list of columns.

        Returns:
            List[Tuple[str, str]]: A list of tuples, each containing the name and description of a column.
        """
        return self._columns

    def generate_prompt(self) -> str:
        """
        Generates a prompt for creating a pandas DataFrame based on number of rows and column information.

        Returns:
            str: A formatted prompt string that includes the DataFrame creation requirements.
        """
        num_columns = len(self._columns)
        columns_info = "\n".join(
            [
                f"  Column name: {col[0]}, type: {col[1]}, description: {col[2] or 'None'}"
                for col in self._columns
            ]
        )

        return PROMPT_TEMPLATE.format(
            num_rows=self._num_rows, num_columns=num_columns, columns_info=columns_info
        )
