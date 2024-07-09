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

from typing import Tuple, TYPE_CHECKING
from collections.abc import Iterator
from collections import deque  # for SchemaContext

from networkx import DiGraph #, bfs_layers, bfs_tree
from google.cloud.bigquery.schema import SchemaField

import bigframes._config as config
from bigframes.core.bqsql_schema_unnest import BQSchemaLayout

if TYPE_CHECKING:
    from bigframes.core.nodes import BigFrameNode


#from functools import partial as ft_partial, reduce as ft_reduce, wraps as ft_wraps
# Schema change lineage/ schema tracking



#TODO: tell whether new cols from join by medata about where the join joins to:root or nested layer?
# then add it to respective leaves/layer in DAG. use join col name as src should be sufficient


# /-- ContextManager section, implemented as command pattern --/
options = config.options

def set_project(project: str | None = None, location: str | None = None,):
    options.bigquery.project = project if project is not None else options.bigquery.project
    options.bigquery.location = location if location is not None else options.bigquery.location
    return

# Command (pattern) interface for schema tracking context manager


# Schema lineage needs a source (join, table, ...) in form of a BigFrameNode or an ArrayValue

