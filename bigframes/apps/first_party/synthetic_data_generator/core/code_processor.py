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

import json
import math
import re

import pandas as pd

from bigframes.ml.llm import PaLM2TextGenerator
import bigframes.pandas as bpd

FLATTENING_SQL_TEMPLATE = """\
WITH B AS (
  SELECT batch FROM ({sf_sql})
),
C AS (
  SELECT batch,
         JSON_VALUE(batch, '$.id') AS id,
         JSON_QUERY_ARRAY(batch, '$.records') AS data
  FROM B
)
SELECT {columns} FROM C, UNNEST(data) as row
"""


class CodeProcessor:
    def __init__(self, prompt: str, col_for_join: bpd.Series, model_connection=None):
        self._model = None
        self._prompt = prompt
        self._col_for_join = col_for_join
        self._connection = model_connection

    def generate_code(self, retries: int = 4):
        last_error = ""
        last_code = ""
        prompt = self._prompt
        code = ""
        temperature = 0
        if self._model is None:
            session = bpd.get_global_session()
            self._model = PaLM2TextGenerator(
                session=session, connection_name=self._connection
            )

        for i in range(retries + 1):
            prompts = bpd.DataFrame(
                {
                    "prompt": [prompt],
                }
            )
            code = (
                self._model.predict(prompts, max_output_tokens=1024, temperature=temperature)
                .to_pandas()
                .iloc[0, 0]
            )

            code = code.replace("```python", "", 1)
            code = code.rsplit("```", 1)[0]
            code = self._ensure_imports(code)

            try:
                self._test_code(code)
                break
            except Exception as e:
                if i == 4:
                    continue

                error = f"{e.__class__.__name__}: {e}"
                if error == last_error and code == last_code:
                    temperature = 0.1
                else:
                    temperature = 0
                last_error = error
                last_code = code
                prompt = self._update_prompt_on_error(error, code)

                print(
                    f"[Info]The generated code has errors, attempting to regenerate.\n error_message = {error}"
                )
        return code

    def update_code_for_num_rows(self, code: str) -> str:
        """
        Modify the code to change the 'num_rows' value and add a comment with original value.

        Args:
            code (str): The original code generated.

        Returns:
            str: The modified code with 'num_rows' value changed and comments added.
        """
        code_lines = code.split("\n")
        modified_value = None
        for idx, line in enumerate(code_lines):
            if "num_rows =" in line:
                current_value = int(line.split("=")[-1].strip())
                modified_value = min(current_value, 100)
                code_lines[
                    idx
                ] = f"num_rows = {modified_value}  # Run: {modified_value}, Submit: {current_value}"
                break

        modified_code = "\n".join(code_lines)

        if modified_value is None:
            warning_comment = (
                "# WARNING: 'num_rows' was not correctly generated.\n"
                "# If the number of rows exceeds 10,000, please try modifying the prompt and regenerating.\n"
            )
            modified_code = warning_comment + modified_code + warning_comment

        return modified_code

    def _test_code(self, code: str) -> pd.DataFrame | None:
        context = (
            {"idx_series": self._col_for_join} if self._col_for_join is not None else {}
        )
        code_lines = code.split("\n")
        found_modified_value = False
        for idx, line in enumerate(code_lines):
            if "num_rows =" in line:
                current_value = int(line.split("=")[-1].strip())
                modified_value = min(current_value, 100)
                code_lines[idx] = f"num_rows = {modified_value}"
                found_modified_value = True
                break

        if not found_modified_value:
            raise Exception(
                "'num_rows should be defined and be used to determine the number of rows of result_df.'"
            )

        test_run_code = "\n".join(code_lines)
        exec(test_run_code, context)

    def run_code(self, code):
        context = (
            {"idx_series": self._col_for_join} if self._col_for_join is not None else {}
        )

        if self._col_for_join is not None:
            code = self._add_extra_col_in_code(code)
        exec(code, context)
        result_df = context.get("result_df", None)
        return result_df

    def submit_code(self, code, max_local_num_rows, rows_per_node):
        submit_value = None
        for line in code.split("\n"):
            if "num_rows" in line and "Submit:" in line:
                submit_value = int(line.split("Submit:")[-1].strip())
                break

        modified_code = re.sub(
            r"num_rows\s*=\s*\d+.*",
            f"num_rows = {rows_per_node if submit_value > max_local_num_rows else submit_value}",
            code,
        )

        modified_code = modified_code + (
            "\nfor column in result_df.columns:\n"
            "    if result_df[column].dtype == 'datetime64[ns]':\n"
            "        result_df[column] = result_df[column].dt.floor('us')"
        )

        if submit_value <= max_local_num_rows:
            context = (
                {"idx_series": self._col_for_join}
                if self._col_for_join is not None
                else {}
            )
            if self._col_for_join is not None:
                modified_code = self._add_extra_col_in_code(modified_code)
            exec(modified_code, context)
            result_df = context.get("result_df", None)
            result_df = bpd.read_pandas(result_df)
        elif self._col_for_join is None:
            result_df = self._remote_function_code_exec(
                modified_code, submit_value, rows_per_node
            )
        else:
            result_df = self._remote_function_code_exec_with_extra_col(
                modified_code, submit_value, rows_per_node
            )

        return result_df

    def _remote_function_code_exec(self, code: str, num_rows: int, rows_per_node: int):
        def get_batch_dyn(id):
            context = {}
            exec(code, context)
            result_df = context.get("result_df", None)
            batch = {
                "id": id,
                "records": json.loads(result_df.to_json(orient="records")),
            }
            return json.dumps(batch)

        lines = code.split("\n")
        libraries = []

        for line in lines:
            if "from " in line:
                libraries += re.findall(r"from (\w+)", line)
            elif "import " in line:
                libraries += re.findall(r"import (\w+)", line)

        unique_libraries = list(set(libraries) - set(["random", "math"]))
        num_nodes = math.ceil(num_rows / rows_per_node)

        # HACK! This file contains the wrapper function code in the GCF source code
        get_batch_dyn.__module__ = "udf.py"

        print("Submit Process Started")
        df = bpd.DataFrame({"id": range(num_nodes)})
        get_batch_remote = bpd.remote_function(
            [int],
            str,
            bigquery_connection=bpd.options.bigquery.bq_connection,
            packages=unique_libraries,
        )(get_batch_dyn)
        sf = df.id.apply(get_batch_remote).rename("batch").to_frame()
        print(f"Deployed GCF {get_batch_remote.bigframes_cloud_function}.")

        record_schema = json.loads(sf.head(1).iloc[0]["batch"])["records"][0].keys()

        columns_clause = ", ".join(
            [
                f"JSON_VALUE(row, '$.{col_name}') AS {col_name}"
                for col_name in record_schema
            ]
        )

        flattening_sql = FLATTENING_SQL_TEMPLATE.format(
            sf_sql=sf.sql, columns=columns_clause
        )

        flattened_df = bpd.read_gbq_query(flattening_sql, max_results=num_rows)
        return flattened_df

    def _remote_function_code_exec_with_extra_col(
        self, code: str, num_rows: int, rows_per_node: int
    ):
        def get_batch_dyn(id):
            context = {}
            exec(code, context)
            result_df = context.get("result_df", None)
            batch = {
                "id": id,
                "records": json.loads(result_df.to_json(orient="records")),
            }
            return json.dumps(batch)

        code = code.replace(
            f"num_rows = {rows_per_node}",
            f"num_rows = random.randint(0,{rows_per_node})",
        )
        code = "import random\n" + code
        lines = code.split("\n")
        libraries = []

        for line in lines:
            if "from " in line:
                libraries += re.findall(r"from (\w+)", line)
            elif "import " in line:
                libraries += re.findall(r"import (\w+)", line)

        unique_libraries = list(set(libraries) - set(["random", "math"]))

        # By average each node will create rows_per_node/2 rows
        # Hence we need to double the row numbers and get some extra
        # To make sure the total number of rows is >= num_rows for
        # the most of time.
        num_nodes = math.ceil(num_rows / rows_per_node * 2.2)

        # HACK! This file contains the wrapper function code in the GCF source code
        get_batch_dyn.__module__ = "udf.py"

        print("Submit Process Started")
        # Calculate the number of sampling needed
        source_df = self._col_for_join.to_frame()
        num_source = len(source_df) / 2  # sample half each time at most.
        num_sample = math.ceil(num_nodes / num_source)
        if num_sample < 10:
            num_sample = 10
            num_source = math.ceil(num_nodes / num_sample)

        samples = [source_df.sample(n=num_source) for _ in range(num_sample)]
        df = bpd.concat(samples, ignore_index=True)
        df = df.head(num_nodes)

        df = df.rename(columns={df.columns[0]: "id"})
        get_batch_remote = bpd.remote_function(
            [str],
            str,
            bigquery_connection=bpd.options.bigquery.bq_connection,
            packages=unique_libraries,
        )(get_batch_dyn)
        sf = df.id.apply(get_batch_remote).rename("batch").to_frame()
        print(f"Deployed GCF {get_batch_remote.bigframes_cloud_function}.")

        record_schema = json.loads(sf.head(1).iloc[0]["batch"])["records"][0].keys()
        columns_clause = ", ".join(
            [
                f"JSON_VALUE(row, '$.{col_name}') AS {col_name}"
                for col_name in record_schema
            ]
        )

        flattening_sql = FLATTENING_SQL_TEMPLATE.format(
            sf_sql=sf.sql, columns=columns_clause
        )
        flattened_df = bpd.read_gbq_query(flattening_sql, max_results=num_rows)
        flattened_df = flattened_df.rename(columns={"id": self._col_for_join.name})

        return flattened_df

    def _add_extra_col_in_code(self, code):
        add_code = (
            "\ncol_name = idx_series.name\n"
            + "result_df[col_name] = np.random.choice(idx_series.values, size=len(result_df))\n"
        )
        return code + add_code

    def _ensure_imports(self, code):
        # Check if numpy and faker are imported
        numpy_imported = "import numpy" in code
        faker_imported = "import faker" in code or "from faker import" in code

        # Add imports if not present
        if not numpy_imported:
            code = "import numpy as np\n" + code
        if not faker_imported:
            code = "from faker import Faker\n" + code

        return code

    def _update_prompt_on_error(self, error: str, code: str) -> str:
        # Handle specific error messages
        if (
            error
            == "Calling `.seed()` on instances is deprecated. Use the class method `Faker.seed()` instead."
        ):
            error = "Calling `.seed()` on instances is deprecated. Use the class method `faker.Faker.seed()` instead."

        updated_prompt = (
            "Generate new code based on the original question, code and the error message of the code shown below\n"
            + "Original Question:\n"
            + self._prompt
            + "\n\n"
            + "Code:\n"
            + code
            + "\n\nError Message: "
            + error
        )
        return updated_prompt
