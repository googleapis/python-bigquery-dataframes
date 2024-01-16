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

from contextlib import contextmanager
from typing import Literal
import warnings

try:
    import IPython
    from IPython.display import display
    import ipywidgets as widgets
except:
    IPython = None

from bigframes.apps.first_party.synthetic_data_generator.core.code_processor import (
    CodeProcessor,
)
from bigframes.apps.first_party.synthetic_data_generator.core.gretel_synthetic_data_engine import (
    GretelSyntheticDataEngine,
)
from bigframes.apps.first_party.synthetic_data_generator.core.prompt_generator import (
    PromptGenerator,
)
from bigframes.apps.first_party.synthetic_data_generator.core.ui import (
    CodeInterface,
    PromptInterface,
    SchemaInterface,
)
from bigframes.apps.first_party.synthetic_data_generator.core.utils import (
    is_notebook,
    summarize_df_to_dict,
)


class SyntheticDataGenerator:
    """
    A class for generating synthetic data based on user-defined schemas or existing dataframes.
    """

    _ROWS_PER_NODE = 100
    _MAX_LOCAL_NUM_ROWS = 10000

    def __init__(
        self,
        model_connection=None,
        model_name: Literal["text-bison", "text-bison-32k"] = "text-bison",
    ) -> None:
        self._interactive = is_notebook()

        self._schema_interface = None
        self._prompt_interface = None
        self._code_interface = None
        self._code_output = None
        self._code_processor = None

        self._model_connection = model_connection
        self._model_name = model_name

    def generate_synthetic_data(self, dataframe_schema_dict=None, interactive=True):
        """
        Generates synthetic data.

        Args:
            dataframe_schema_dict (Optional[Dict[str, Any]]):
                A dictionary defining the schema of the dataframe to be generated.
                It should have two keys: 'num_rows' for the number of rows, and 'columns'
                which is a list of tuples. Each tuple should contain three elements:
                column name, column data type, and column description.
                Example: {'num_rows': 100, 'columns': [('name', 'str', 'Name of the person'), ...]}
                Note: Some data types might not be supported and could cause issues.
            interactive (bool):
                If True, enables interactive mode for data generation, allowing user input for schema definition.
                Defaults to True. Automatically set to False if not running in a Jupyter notebook environment.
        """
        self._interactive = is_notebook() and interactive

        if not self._interactive and dataframe_schema_dict is None:
            raise ValueError(
                "dataframe_schema_dict cannot be empty in non-interactive mode"
            )

        self._generate_dataframe(description=dataframe_schema_dict)

    def generate_synthetic_data_from_table(
        self,
        orig_df,
        num_rows=None,
        modify_schema=False,
        engine="PaLM2",
        use_string_column_values=False,
        interactive=True,
    ):
        """
        Generates synthetic data based on an original dataframe.

        Args:
            orig_df (pd.DataFrame):
                The original dataframe to base the synthetic data on.
            num_rows (Optional[int]):
                The number of rows for the synthetic data. Defaults to 1000 if not provided.
            modify_schema (bool):
                If True, allows modification of the schema. Default is False.
            engine (str):
                The engine to use for generating data. Defaults to 'PaLM2', choices are ['PaLM2', 'Gretel'].
            use_string_column_values (bool):
                If True, uses sample string values for the columns generation.
            interactive (bool):
                If True, enables interactive mode for data generation, allowing user input for schema definition.
                Defaults to True. Automatically set to False if not running in a Jupyter notebook environment.
        """
        engine = engine.lower()
        self._interactive = is_notebook() and interactive

        if engine == "palm2":
            self._generate_dataframe(
                orig_df=orig_df,
                num_rows=num_rows,
                modify_schema=modify_schema,
                use_string_column_values=use_string_column_values,
            )
        elif engine == "gretel":
            data_generator = GretelSyntheticDataEngine()
            self.generated_df = data_generator.train_and_generate_synthetic_data(
                orig_df=orig_df, num_rows=num_rows
            )
        else:
            raise ValueError(f"Unsupported engine: {engine}")

    def generate_synthetic_correlated_data(
        self, correlated_column, dataframe_schema_dict=None, interactive=True
    ):
        """
        Generates synthetic data that contains elements from a specific column.

        Args:
            correlated_column (pd.Series):
                The column to correlate the synthetic data with.
            dataframe_schema_dict (Optional[Dict[str, Any]]):
                A dictionary defining the schema of the dataframe to be generated.
                It should have two keys: 'num_rows' for the number of rows, and 'columns'
                which is a list of tuples. Each tuple should contain three elements:
                column name, column data type, and column description.
                Example: {'num_rows': 100, 'columns': [('name', 'str', 'Name of the person'), ...]}
                Note: Some data types might not be supported and could cause issues.
            interactive (bool):
                If True, enables interactive mode for data generation, allowing user input for schema definition.
                Defaults to True. Automatically set to False if not running in a Jupyter notebook environment.
        """
        self._interactive = is_notebook() and interactive
        if not self._interactive and dataframe_schema_dict is None:
            raise ValueError(
                "dataframe_schema_dict cannot be empty in non-interactive mode"
            )
        self._generate_dataframe(
            description=dataframe_schema_dict, col_for_join=correlated_column
        )

    def generate_synthetic_correlated_data_from_table(
        self,
        orig_df,
        num_rows=None,
        modify_schema=False,
        correlated_column=None,
        use_string_column_values=False,
        interactive=True,
    ):
        """
        Generates synthetic data based on an original dataframe and correlates it with a specific column.

        Args:
            orig_df (pd.DataFrame):
                The original dataframe to base the synthetic data on.
            num_rows (Optional[int]):
                The number of rows for the synthetic data. Defaults to 1000 if not provided.
            modify_schema (bool):
                If True, allows modification of the schema. Default is False.
            correlated_column (Optional[pd.Series]):
                The column to correlate the synthetic data with. Must be provided.
            use_string_column_values (bool):
                If True, uses sample string values for the columns generation.
            interactive (bool):
                If True, enables interactive mode for data generation, allowing user input for schema definition.
                Defaults to True. Automatically set to False if not running in a Jupyter notebook environment.
        """
        if correlated_column is None:
            raise ValueError("Please provide a correlated column")

        self._interactive = is_notebook() and interactive
        self._generate_dataframe(
            orig_df=orig_df,
            num_rows=num_rows,
            modify_schema=modify_schema,
            col_for_join=correlated_column,
            use_string_column_values=use_string_column_values,
        )

    def _generate_dataframe(
        self,
        orig_df=None,
        description=None,
        num_rows=None,
        modify_schema=False,
        col_for_join=None,
        use_string_column_values=False,
    ):
        self.output = widgets.Output()
        self.col_for_join = col_for_join

        self.generated_df = None

        if orig_df is not None:
            self._init_with_dataframe(
                orig_df,
                num_rows,
                modify_schema,
                use_string_column_values=use_string_column_values,
            )
        else:
            self._display_set_schema_interface(
                description=description, modify_schema=modify_schema
            )

    def _init_with_dataframe(
        self, df, num_rows, modify_schema, use_string_column_values=False
    ):
        if num_rows is None:
            num_rows = 1000
            warnings.warn("No 'num_rows' provided, defaulting to 1000.")
        if self._interactive:
            output = widgets.Output()
            display(output)

            with output:
                description = summarize_df_to_dict(
                    df, num_rows, use_string_column_values=use_string_column_values
                )
            output.close()
        else:
            description = summarize_df_to_dict(
                df, num_rows, use_string_column_values=use_string_column_values
            )

        self._display_set_schema_interface(
            description=description, modify_schema=modify_schema
        )

    def _display_set_schema_interface(self, description=None, modify_schema=False):
        if self._interactive:
            self._schema_interface = SchemaInterface(
                on_submit_callback=self._on_schema_submit_callback
            )
            self._schema_interface.display_interface(description)
            self.columns = []
            if not modify_schema and description is not None:
                prompts = PromptGenerator(description).generate_prompts()
                self._display_prompt_interface(prompts)
        elif description is None:
            raise ValueError("No schema description provided for non-interactive mode.")
        else:
            prompts = PromptGenerator(description).generate_prompts()
            self._display_prompt_interface(prompts)

    def _display_prompt_interface(self, prompts):
        if self._interactive:
            self._prompt_interface = PromptInterface(
                on_submit_callback=self._on_prompt_submit_callback
            )
            self._prompt_interface.display_interface(prompts=prompts)
        self._display_code_interface(prompts)

    def _disable_all_buttons(self):
        if self._schema_interface is not None:
            self._schema_interface.disable_inputs()
        if self._prompt_interface is not None:
            self._prompt_interface.disable_inputs()
        if self._code_interface is not None:
            self._code_interface.disable_inputs()

    def _enable_all_buttons(self):
        if self._schema_interface is not None:
            self._schema_interface.enable_inputs()
        if self._prompt_interface is not None:
            self._prompt_interface.enable_inputs()
        if self._code_interface is not None:
            self._code_interface.enable_inputs()

    @contextmanager
    def _disable_buttons_temporarily(self):
        self._disable_all_buttons()
        try:
            yield
        finally:
            self._enable_all_buttons()

    def _display_code_interface(self, prompts):
        self._code_processor = CodeProcessor(
            prompts,
            self.col_for_join,
            model_connection=self._model_connection,
            model_name=self._model_name,
        )
        codes = ""
        if self._interactive:
            with self._disable_buttons_temporarily():
                prompt_output = widgets.Output()
                display(prompt_output)
                with prompt_output:
                    codes = self._code_processor.generate_codes()
                prompt_output.close()

                self._code_output = widgets.Output()
                codes = self._code_processor.update_codes_for_num_rows(codes)
                self._code_interface = CodeInterface(
                    on_run_callback=self._on_code_run_callback,
                    on_submit_callback=self._on_code_submit_callback,
                )
                self._code_interface.display_interface(codes)
                display(self._code_output)
        else:
            codes = self._code_processor.generate_codes(interactive=False)
            codes = self._code_processor.update_codes_for_num_rows(codes)
            self.generated_df = self._code_processor.submit_codes(
                codes, self._MAX_LOCAL_NUM_ROWS, self._ROWS_PER_NODE
            )
            print("Submitted, now you should have access to the dataframe.")

    def _on_schema_submit_callback(self, description):
        if self._prompt_interface is not None:
            self._prompt_interface.close_interface()
            self._prompt_interface = NotImplementedError
        if self._code_interface is not None:
            self._code_interface.close_interface()
            self._code_interface = None
            self._code_output.close()
            self._code_output = None

        prompt = PromptGenerator(description).generate_prompts()
        self._display_prompt_interface(prompt)

    def _on_prompt_submit_callback(self, prompts):
        if self._code_interface is not None:
            self._code_interface.close_interface()
            self._code_interface = None
            self._code_output.close()
            self._code_output = None

        self._display_code_interface(prompts)

    def _on_code_run_callback(self, codes):
        if self._code_processor is not None:
            with self._disable_buttons_temporarily():
                self._code_output.clear_output(wait=True)
                with self._code_output:
                    result_df = self._code_processor.run_codes(codes)
                    print(result_df)

    def _on_code_submit_callback(self, codes):
        if self._code_processor is not None:
            with self._disable_buttons_temporarily():
                self._code_output.clear_output(wait=True)
                with self._code_output:
                    self.generated_df = self._code_processor.submit_codes(
                        codes, self._MAX_LOCAL_NUM_ROWS, self._ROWS_PER_NODE
                    )
                    print("Submitted, now you should have access to the dataframe.")
