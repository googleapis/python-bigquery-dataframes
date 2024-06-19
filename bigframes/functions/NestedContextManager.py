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

from abc import ABC, abstractmethod
from networkx import DiGraph, topological_sort

from bigframes.dataframe import DataFrame
from bigframes.series import Series
from bigframes.functions.nested_utils import MemberSelector
from bigframes.functions.nested import BQSchemaLayout
from google.cloud.bigquery.schema import SchemaField
from google.cloud import bigquery
from typing import List, Callable, ContextManager
from contextvars import ContextVar
import bigframes.pandas as bfpd
import ast
from collections import defaultdict, deque
from functools import partial as ft_partial, reduce as ft_reduce, wraps as ft_wraps


#TODO: tell whether new cols from join by medata about where the join joins to:root or nested layer?
# then add it to respective leaves/layer in DAG. use join col name as src should be sufficient



class NodeInfo(MemberSelector):
    T_root: str = "ROOT"
    T_record: str = "RECORD"

    def __init__(self, node_type: str, children: List[str]):
        self.node_type = node_type
        self.children = children


class SchemaHandler:
    def __init__(self, schema: BQSchemaLayout, layer_separator: str, struct_separator: str):
        self.schema_orig: BQSchemaLayout = schema
        self.sep_layers = layer_separator  # TODO: remove? needed at all here? Probably for DAG class..
        self.sep_structs = struct_separator

    def _tree_from_strings(self, paths: List[str]):
        # we need a root 'node' root, otherwise we would have to know all names in the first layer later
        root = {} # defaultdict(dict)?
        for path in paths:
            parts = path.split(self.sep_structs)
            node = root
            for part in parts:
                node = node.setdefault(part, {})
        return root

    def bfs(self, tree: List[str], root: str | None=None):
        # bfs on "."-joined strings
        # start queue with root key-value pair
        queue = deque([[root]]) if root is not None else deque([([], tree)])

        while queue:
            layer = []
            for _ in range(len(queue)):
                # get current item and traverse its direct succesors
                path, node = queue.popleft()
                for key, child in node.items():
                    # build key string by concatenating with path/ predecessor's name
                    new_path = path + self.sep_structs + key if path else key
                    # add item to layer/ current level in tree
                    layer.append(new_path)
                    # append item to queue for breadth first search
                    queue.append((new_path, child))
            if layer:
                yield layer

    def orig_schema(self) -> dict:
        col_tree = self._tree_from_strings(list(self.schema_orig.map_to_list.keys()))
        return col_tree



# /-- ContextManager section, implemented as command pattern --/

# Command interface
class CommandBase(ABC):
    def __init__(self, receiver: SchemaHandler):
        self.receiver = receiver
        
    @abstractmethod
    def execute(self):
        ...

def set_project(project: str | None = None, location: str | None = None,):
    bfpd.options.bigquery.project = project if project is not None else bfpd.options.bigquery.project
    bfpd.options.bigquery.location = location if location is not None else bfpd.options.bigquery.loca
    return

class NestedDataFrame(CommandBase):
    _default_sep_layers: str = "__"  # TODO: make sure it is not found in column names!
    _sep_structs: str = "."  # not ot be modified by user
    _base_root_name = "_root_"

    # setup, start schema deduction
    def __init__(self, data: DataFrame | Series | str, layer_separator: str | None = None):
        # TODO: change into parameters
        # this needs to be done before getting the schema
        assert(bfpd.options.bigquery.project is not None and bfpd.options.bigquery.location is not None)

        # will be frequently used to get schemata
        self.client = bigquery.Client(project=bfpd.options.bigquery.project)
        
        # get schema information
        self._is_nested = True  # set before calling _deduct_schema as it is used there
        _schema = self._deduct_schema(data, NestedDataFrame._sep_structs)

        # set receiver and build DAG from receiver and its schema information
        layer_separator = layer_separator if layer_separator is not None else NestedDataFrame._default_sep_layers
        receiver = SchemaHandler(_schema, layer_separator=layer_separator, struct_separator=NestedDataFrame._sep_structs)
        super().__init__(receiver)
        self.dag = DiGraph()
        self.dag.add_node(self._base_root_name, node_type=NodeInfo.T_root)  # ONE common root note as multiple columns can follow
        self._dag_from_schema()

    def _query_table_schema(self, data: DataFrame | Series | str) -> list | list[SchemaField]:
        table_ref = data.to_gbq(destination_table=None,  ordering_id="_bf_idx_id") \
            if not isinstance(data, str) else data
            # project_id=bfpd.options.bigquery.project

        query_job = self.client.get_table(table_ref)
        schema = query_job.schema
        self._is_nested = NestedDataFrame._has_nested_data(schema)
        return schema

    def _deduct_schema(self, data: DataFrame | Series | str, struct_separator: str) -> BQSchemaLayout:
        if isinstance(data, str):
            self._current_data = bfpd.read_gbq(data)
            _data = self._current_data if "SELECT" in data.upper() else data
        else:
            self._current_data = data
            _data = data
        schema = self._query_table_schema(_data)
        if schema:
            schema_layout = BQSchemaLayout(schema)
            schema_layout.determine_layout(struct_separator)
            return schema_layout
        return None

    # Context Manager interface
    def __enter__(self):
        assert(self.receiver.schema_orig is not None)
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.df = None
        return

    # Context Manager functionality: Overload |=

    # # data or function right after |= assignment
    # def __ror__(self, other: DataFrame | Series):
    #     return self.execute(other)

    # no data, just function after | pipe
    def __or__(self, other: DataFrame | Series):
        return self.execute(other)

    # Private helper methods for starting schema deduction and DAG creation
    @staticmethod
    def _has_nested_data(schema: list) -> dict | None:
        return sum([1 for x in schema if x.field_type == "RECORD"]) > 0

    def _dag_from_schema(self):
        schema = self.receiver.orig_schema()
        
        root_layer = True
        for layer in self.receiver.bfs(schema):
            for col_name in layer:
                assert(self.receiver.sep_layers not in col_name)
                if not root_layer:
                    col_type = self.receiver.schema_orig.map_to_type[col_name]  # RECORD?
                    parent = col_name.rsplit(self.receiver.sep_structs, 1)[0]
                else:
                    parent = self._base_root_name
                    col_type = NodeInfo.T_root
                # replace struct separator with layer separator, as struct separator must not be used in exploded column names
                col_name = col_name.replace(self.receiver.sep_structs, self.receiver.sep_layers)
                self.dag.add_node(col_name, node_type=col_type)
                self.dag.add_edge(parent, col_name)
            root_layer = False
        return
    
    def _value_multiplicities(self, input_dict: dict[str, List[str]]) -> dict[tuple, str]:
        """
        Finds multiplicities of values in a dict and returns an 'inverse' dict with keys as value list.
        Changing the key from list to hashable tuple we can cover two cases in once:
            a) single column explodes into multiple others, such as OneHotEncoding
            b) multiple columns merge into a single one, such as Merge for categories with small amount of samples
        :param dict[str, List[str]] input_dict: dict with keys as column names and values as list of column names
        :return: dict with keys as value list and values as list of column names
        """
        inverted_dict = {}
        for key, value in input_dict.items():

            value_tuple = tuple(sorted(value))
            inverted_dict.setdefault(value_tuple, []).append(key)
        duplicates = {value: keys for value, keys in inverted_dict.items() if len(keys) > 1}
        #TODO: add NodeInfo?
        return duplicates

    # -- user interface/ public methods and command interface --

    @property
    def data(self):
        return self._current_data

    @property
    def is_nested(self):
        return self._is_nested

    #def execute(self, *args, actions: Callable | List[Callable] | None = None, **kwargs):
    def execute(self, data: DataFrame | Series):
        schema_changes = {}
         #TODO: get schema changes!
        successors = self._value_multiplicities(schema_changes)
        #TODO: add to dag and handle potential joins as described at top of this file
        #TODO: changes = ContextVar("changes", default={})
        #  changes.set(value)
        #schema = self._query_table_schema(data)
        return successors  #TODO: change
