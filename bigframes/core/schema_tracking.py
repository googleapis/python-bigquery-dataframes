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

from typing import Tuple #, TYPE_CHECKING
from collections.abc import Iterator
from collections import deque  # for SchemaContext

from networkx import DiGraph #, bfs_layers, bfs_tree
from google.cloud.bigquery.schema import SchemaField

import bigframes.pandas as bfpd
from bigframes.core.bqsql_schema_unnest import BQSchemaLayout

# if TYPE_CHECKING:
#     from bigframes.core.nodes import BigFrameNode

#from functools import partial as ft_partial, reduce as ft_reduce, wraps as ft_wraps
# Schema change lineage/ schema tracking



#TODO: tell whether new cols from join by medata about where the join joins to:root or nested layer?
# then add it to respective leaves/layer in DAG. use join col name as src should be sufficient


# /-- ContextManager section, implemented as command pattern --/

def set_project(project: str | None = None, location: str | None = None,):
    bfpd.options.bigquery.project = project if project is not None else bfpd.options.bigquery.project
    bfpd.options.bigquery.location = location if location is not None else bfpd.options.bigquery.loca
    return

# Command (pattern) interface for schema tracking context manager


# Schema lineage needs a source (join, table, ...) in form of a BigFrameNode
class SchemaSource:
    def __init__(self, dag: DiGraph, schema_orig: Tuple[SchemaField, ...], schema_bq: BQSchemaLayout) -> None:
        self.schema_orig = schema_orig
        self.schema_bq = schema_bq
        self.dag = dag

    @property
    def is_valid(self) -> bool:
        """
        Returns True if self._dag is not None, which is the case whenever the BigFrameNode has a physical_schema attribute.
        Other cases will be handled in the near future.
        """
        return self.dag is not None


class SchemaSourceHandler:
    _base_root_name = "_root_"

    def __init__(self):
        self._sources = {}

    @property
    def sources(self) -> dict:
        return self._sources

    @staticmethod
    def _tree_from_strings(paths: list[str], struct_separator: str) -> dict:
        root = {}
        for path in paths:
            parts = path.split(struct_separator)
            node = root
            for part in parts:
                node = node.setdefault(part, {})
        return root
    
    @staticmethod
    def bfs(tree: list[str], layer_separator: str, root: str|None=None) -> Iterator[list]:
        """
        Iteraror function for BQ schema base on Tuple[SchemaField, ...]. Returns layer using breadth first search.
        """
        # bfs on "."-joined strings, the "." is the struct_separator
        # start queue with root key-value pair
        queue = deque([[root]]) if root is not None else deque([([], tree)])

        while queue:
            layer = []
            for _ in range(len(queue)):
                # get current item and traverse its direct succesors
                path, node = queue.popleft()
                for key, child in node.items():
                    # build key string by concatenating with path/ predecessor's name
                    new_path = path + layer_separator + key if path else key
                    # add item to layer/ current level in tree
                    layer.append(new_path)
                    # append item to queue for breadth first search
                    queue.append((new_path, child))
            if layer:
                yield layer

    def _init_dag_from_schema(self, dag: DiGraph, schema: BQSchemaLayout, layer_separator: str, struct_separator: str) -> DiGraph:
        root_layer = True
        dag_ret = dag
        bq_schema = self._tree_from_strings(list(schema.map_to_list.keys()), struct_separator)
        for layer in self.bfs(bq_schema, layer_separator=layer_separator):
            for col_name in layer:
                assert(layer_separator not in col_name)
                last_layer = col_name.rsplit(struct_separator, 1)[0] if not root_layer else self._base_root_name
                col_type = schema.map_to_type[col_name]
                # replace struct separator with layer separator, as struct separator must not be used in exploded column names
                col_name = col_name.replace(struct_separator, layer_separator)
                dag_ret.add_node(col_name, node_type=col_type)
                dag_ret.add_edge(last_layer, col_name)
            root_layer = False
        return dag_ret

    def _dag_to_schema(self):
        # layers = bfs_layers(self._dag, self._base_root_name)
        # bfs = bfs_tree(self._dag, self._base_root_name)
        # parent_layer = self._base_root_name
        pass

    def add_source(self, src: 'BigFrameNode', layer_separator: str, struct_separator: str) -> None:
        """Adds new SchemaSource for src to self._sources"""
        schema_orig: Tuple[SchemaField, ...] = src.physical_schema if hasattr(src, "physical_schema") else None
        schema = None
        dag = None
        if schema_orig:
            schema = BQSchemaLayout(schema_orig)
            schema.determine_layout(struct_separator)
            dag = DiGraph()
            # ONE common root note as multiple columns can follow
            dag.add_node(self._base_root_name, node_type=self._base_root_name)
            dag = self._init_dag_from_schema(dag, schema, layer_separator, struct_separator) 
        source = SchemaSource(dag, schema_orig, schema)
        self._sources[src] = source

    def _value_multiplicities(self, input_dict: dict[str, list[str]]) -> dict[tuple, str]:
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
    
    def exists(self, src: 'BigFrameNode') -> SchemaSource|None:
        """Returns SchemaSource if src exists, else None."""
        return self._sources.get(src, None)
    
    

class SchemaTrackingContextManager:
    """
    Context manager for schema tracking using command pattern.
    Utilizes a DAG for schema lineage and thus can reconstruct each step of schema changes.
    """
    _default_sep_layers: str = "__"  # make sure it is not a substring of any column name!
    _default_sep_structs: str = "."  # not ot be modified by user
    _is_active = False

    # setup, start schema deduction
    #def __init__(self, data: DataFrame | Series | str | None=None, layer_separator: str | None = None):
    def __init__(self, layer_separator: str | None = None, struct_separator: str | None = None):
        # TODO: change into parameters
        # this needs to be done before getting the schema
        assert(bfpd.options.bigquery.project is not None and bfpd.options.bigquery.location is not None)
        self.sep_layers = layer_separator if layer_separator is not None else SchemaTrackingContextManager._default_sep_layers
        self.set_structs = struct_separator if struct_separator is not None else SchemaTrackingContextManager._default_sep_structs
        self._source_handler = SchemaSourceHandler()

    @property
    @staticmethod
    def active():
        """
        Returns True if context manager is active, ie if we are within a "with" block
        """
        return SchemaTrackingContextManager._is_active

    def add_source(self, src: 'BigFrameNode') -> None:
        """Adds new SchemaSource for src to self._sources. Key is src."""
        if self._source_handler.exists(src) is not None:
            raise ValueError(f"{self.__class__.__name__}:{self.__class__.__qualname__}: Source {src} already exists")
        self._source_handler.add_source(src, layer_separator=self.sep_layers, struct_separator=self.set_structs)

    # Context Manager interface
    def __enter__(self):
        SchemaTrackingContextManager._is_active = True
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        #self._dag_to_schema()
        SchemaTrackingContextManager._is_active = False
        #TODO: compute final schema from DAG
        return

    # Private helper methods for starting schema deduction and DAG creation
    @staticmethod
    def _has_nested_data(schema: list) -> dict | None:
        return sum([1 for x in schema if x.field_type == "RECORD"]) > 0


NestedDataContextManager = SchemaTrackingContextManager()
