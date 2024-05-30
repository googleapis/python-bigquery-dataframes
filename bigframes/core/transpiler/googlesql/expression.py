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

import dataclasses

import google.cloud.bigquery as bigquery

import bigframes.core.transpiler.googlesql.sql as sql


@dataclasses.dataclass
class AsAlias(sql.SQLSyntax):
    alias: str

    def sql(self) -> str:
        return f"AS {self.alias}"


@dataclasses.dataclass
class ABCExpression(sql.SQLSyntax):
    pass


@dataclasses.dataclass
class Expression(ABCExpression):
    name: str

    def sql(self) -> str:
        # https://cloud.google.com/bigquery/docs/reference/standard-sql/lexical#identifiers
        # Just always escape, otherwise need to check against every reserved sql keyword
        return f"`{self._escape_special_characters(self.name)}`"

    @staticmethod
    def _escape_special_characters(value: str):
        """Escapes all special charactesrs:
        https://cloud.google.com/bigquery/docs/reference/standard-sql/lexical#string_and_bytes_literals
        """
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


@dataclasses.dataclass
class StarExpression(ABCExpression):
    def sql(self) -> str:
        return "*"


@dataclasses.dataclass
class TableRef(ABCExpression):
    table_ref: bigquery.TableReference

    def sql(self) -> str:
        return f"`{self.table_ref.project}`.`{self.table_ref.dataset_id}`.`{self.table_ref.table_id}`"
