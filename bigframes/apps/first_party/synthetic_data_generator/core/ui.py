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

from IPython.display import display
import ipywidgets as widgets

# TODO(huanc): many shared functions, can have one base class.


class SchemaInterface:
    def __init__(self, on_submit_callback=None):
        self.columns_box = widgets.VBox([])
        self.row_input = widgets.IntText(
            value=100,
            description="Number of Rows:",
            disabled=False,
            layout={"width": "200px"},
            style={"description_width": "100px"},
        )
        self.add_new_row_button = widgets.Button(description="Add new column")
        self.add_new_row_button.on_click(self._add_new_row)
        self.submit_button = widgets.Button(description="Submit Request")
        self.submit_button.on_click(self._on_submit_button_clicked)
        self.layout = widgets.VBox(
            [
                self.row_input,
                self.columns_box,
                self.add_new_row_button,
                self.submit_button,
            ]
        )

        self._on_submit_callback = on_submit_callback

    def display_interface(self, description: dict = None):
        if description is not None:
            self.row_input.value = description["num_rows"]
            for col_name, col_type, col_desc in description["columns"]:
                self._add_new_row(
                    column_name=col_name,
                    column_type=col_type,
                    column_description=col_desc,
                )
        else:
            self._add_new_row()
        display(self.layout)
        self.enable_inputs()

    def _add_new_row(
        self, b=None, column_name="", column_type="Int64", column_description=""
    ):
        name_input = widgets.Text(
            value=column_name,
            placeholder="Column name",
            description="Column Name:",
            style={"description_width": "100px"},
        )
        dtype_dropdown = widgets.Dropdown(
            options=[
                "Int64",
                "Float64",
                "boolean",
                "string",
                "pd.Timestamp UTC",
                "pd.Timestamp",
            ],
            value=column_type,
            description="Column DType:",
            style={"description_width": "100px"},
        )
        description_textarea = widgets.Textarea(
            value=column_description,
            placeholder="Type description here...",
            description="Column Description:",
            layout={"height": "100px", "width": "100%"},
            style={"description_width": "150px"},
        )
        delete_button = widgets.Button(description="Delete", icon="trash")
        delete_button.on_click(lambda btn: self._delete_row(btn, col_box))

        col_box = widgets.HBox(
            [name_input, dtype_dropdown, description_textarea, delete_button]
        )
        self.columns_box.children = self.columns_box.children + (col_box,)

    def _delete_row(self, btn, col_box):
        self.columns_box.children = tuple(
            child for child in self.columns_box.children if child != col_box
        )

    def _on_submit_button_clicked(self, btn):
        if self._on_submit_callback is not None:
            self._on_submit_callback(self._input_to_prompt())
        else:
            print("Submit button clicked, but no callback is set.")

    def disable_inputs(self):
        """Disable all input fields and buttons."""
        self.row_input.disabled = True
        self.submit_button.disabled = True
        self.add_new_row_button.disabled = True
        for child in self.columns_box.children:
            for widget in child.children:
                if isinstance(widget, widgets.Widget):
                    widget.disabled = True

    def enable_inputs(self):
        """Enable all input fields and buttons."""
        self.row_input.disabled = False
        self.submit_button.disabled = False
        self.add_new_row_button.disabled = False
        for child in self.columns_box.children:
            for widget in child.children:
                if isinstance(widget, widgets.Widget):
                    widget.disabled = False

    def _input_to_prompt(self):
        num_rows = self.row_input.value
        columns = []
        # Extract column types and descriptions
        for idx, child in enumerate(self.columns_box.children):
            name_input, dropdown, textarea, _ = child.children
            column_name = (
                name_input.value if name_input.value.strip() != "" else f"column_{idx}"
            )
            column_dtype = dropdown.value
            column_description = (
                textarea.value if textarea.value.strip() != "" else "None"
            )

            columns.append((column_name, column_dtype, column_description))

        description = {"num_rows": num_rows, "columns": columns}
        return description


class PromptInterface:
    def __init__(self, on_submit_callback=None):
        self.submit_prompts_button = widgets.Button(description="Submit Prompts")
        self.submit_prompts_button.on_click(self._on_submit_button_clicked)
        self._on_submit_callback = on_submit_callback
        self.prompts_textareas = []
        self.layout = widgets.VBox([])

    def display_interface(self, prompts):
        """
        Create and display the prompt interface with the given prompts.
        """
        self.prompts_textareas = [
            widgets.Textarea(
                value=prompt,
                placeholder=f"Type your prompt here...",
                description=f"Prompt {i+1}:",
                layout={"height": "100px", "width": "100%"},
                style={"description_width": "100px"},
            )
            for i, prompt in enumerate(prompts)
        ]
        self.layout.children = self.prompts_textareas + [self.submit_prompts_button]
        display(self.layout)

    def _on_submit_button_clicked(self, btn):
        if self._on_submit_callback is not None:
            self._on_submit_callback(
                [textarea.value for textarea in self.prompts_textareas]
            )
        else:
            print("Submit button clicked, but no callback is set.")

    def disable_inputs(self):
        """
        Disable all input fields and buttons in the interface.
        """
        for textarea in self.prompts_textareas:
            textarea.disabled = True
        self.submit_prompts_button.disabled = True

    def enable_inputs(self):
        """
        Enable all input fields and buttons in the interface.
        """
        for textarea in self.prompts_textareas:
            textarea.disabled = False
        self.submit_prompts_button.disabled = False

    def close_interface(self):
        """
        Close and remove the prompts interface from display.
        """
        self.layout.close()


class CodeInterface:
    def __init__(
        self,
        col_for_join=None,
        modified_value=None,
        on_run_callback=None,
        on_submit_callback=None,
    ):
        self.col_for_join = col_for_join
        self.modified_value = modified_value
        self._on_run_callback = on_run_callback
        self._on_submit_callback = on_submit_callback

        self.run_button = widgets.Button(description="Run Code")
        self.submit_button = widgets.Button(description="Submit Code")

        self.run_button.on_click(self._on_run_button_clicked)
        self.submit_button.on_click(self._on_submit_button_clicked)

        self.codes_textareas = []
        self.layout = widgets.VBox([])

    def display_interface(self, codes):
        """
        Dynamically create and display the code interface for each code snippet.
        """
        self.codes_textareas = [
            widgets.Textarea(
                value=code,
                placeholder="Type your code here...",
                description=f"Code {i+1}:",
                layout={"height": "200px", "width": "100%"},
                style={"description_width": "50px"},
            )
            for i, code in enumerate(codes)
        ]

        self.layout.children = self.codes_textareas + [
            self.run_button,
            self.submit_button,
        ]
        display(self.layout)

    def _on_run_button_clicked(self, btn):
        if self._on_run_callback is not None:
            codes = [textarea.value for textarea in self.codes_textareas]
            self._on_run_callback(codes)
        else:
            print("Run button clicked, but no callback is set.")

    def _on_submit_button_clicked(self, btn):
        if self._on_submit_callback is not None:
            codes = [textarea.value for textarea in self.codes_textareas]
            self._on_submit_callback(codes)
        else:
            print("Submit button clicked, but no callback is set.")

    def close_interface(self):
        """
        Close and remove the code interface from display.
        """
        self.layout.close()

    def disable_inputs(self):
        """
        Disable all input fields and buttons in the code interface.
        """
        for textarea in self.codes_textareas:
            textarea.disabled = True
        self.run_button.disabled = True
        self.submit_button.disabled = True

    def enable_inputs(self):
        """
        Enable all input fields and buttons in the code interface.
        """
        for textarea in self.codes_textareas:
            textarea.disabled = False
        self.run_button.disabled = False
        self.submit_button.disabled = False
