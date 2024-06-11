# Copyright 2024 Google LLC
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


def _escape_special_characters(value: str):
    """Escapes all special charactesrs"""
    # https://cloud.google.com/bigquery/docs/reference/standard-sql/lexical#string_and_bytes_literals
    trans_table = str.maketrans(
        {
            "\a": r"\a",
            "\b": r"\b",
            "\f": r"\f",
            "\n": r"\n",
            "\r": r"\r",
            "\t": r"\t",
            "\v": r"\v",
            "\\": r"\\",
            "?": r"\?",
            '"': r"\"",
            "'": r"\'",
            "`": r"\`",
        }
    )
    return value.translate(trans_table)
