from __future__ import annotations

import bigframes._config as config
import bigframes.core.blocks as blocks
import bigframes.core.nodes as nodes
from bigframes.dtypes import is_struct_like as dt_is_struct_like

from pyarrow import DataType as pa_datatype
from networkx import DiGraph, topological_sort
from google.cloud.bigquery.schema import SchemaField
from typing import final, Tuple, TYPE_CHECKING
from collections.abc import Mapping, Callable


if TYPE_CHECKING:
    from bigframes.dataframe import DataFrame


# -- schema tracking --
def bfnode_hash(node: nodes.BigFrameNode):
    return node._node_hash
    

class SchemaSource:
    def __init__(self, node_flattened: nodes.BigFrameNode, dag: DiGraph, 
                 schema: dict, schema_orig: Tuple[SchemaField, ...] | None=None,
                 ) -> None:
        self.node_flattened = node_flattened
        self.dag = dag
        self.schema = schema
        self.schema_orig = schema_orig
        
    @property
    def is_valid(self) -> bool:
        """
        Returns True if self._dag is not None, which is the case whenever the ArrayValue's BigFrameNode has a physical_schema attribute.
        Other cases will be handled in the near future.
        """
        return self.dag is not None


class SchemaSourceHandler:
    _base_root_name = "_root_"

    def __init__(self):
        self._sources = {}
        self._order = []

    @property
    def sources(self) -> dict:
        return self._sources

    @property
    def order(self) -> list:
        return self._order

    @staticmethod
    def _tree_from_strings(paths: list[str], struct_separator: str) -> dict:
        root = {}
        for path in paths:
            parts = path.split(struct_separator)
            node = root
            for part in parts:
                node = node.setdefault(part, {})
        return root
    
    def _init_dag_from_df_schema(self, dag: DiGraph, schema: dict[str, tuple], layer_separator: str, struct_separator: str) -> DiGraph:
        dag_ret = dag
        root = [el for el in topological_sort(dag)][0]

        for key, value in schema.items():
            parent = value[0] if value[0] else root
            dag_ret.add_node(key, node_type=schema[key])
            dag_ret.add_edge(parent, key)

        #TODO: Debug log info
        #print([el for el in topological_sort(dag)])
        return dag_ret
    
    @staticmethod
    def leafs(dag: DiGraph):
        return [node for node in dag.nodes if dag.out_degree(node) == 0]

    # two identical properties, depending on what meaning you prefer
    def _dag_to_schema(self):
        # layers = bfs_layers(self._dag, self._base_root_name)
        # bfs = bfs_tree(self._dag, self._base_root_name)
        # parent_layer = self._base_root_name
        pass

    def dag_from_df(self, schema: dict[str, tuple],
                   struct_separator: str, layer_separator: str) -> DiGraph:
        dag = DiGraph()
        dag.add_node(self._base_root_name, node_type=self._base_root_name)
        #dag_dict = self._tree_from_strings(cols_flattened, struct_separator=layer_separator)
        dag_res = self._init_dag_from_df_schema(dag, schema, layer_separator=layer_separator, struct_separator=struct_separator)
        return dag_res
    
    def add_source(self, hash_df: int, source:SchemaSource):
        self._sources[hash_df] = source
        self._order.append(hash_df)

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

    def get(self, src: nodes.BigFrameNode) -> SchemaSource|None:
        """Returns SchemaSource if src exists, else None."""
        node_hash = bfnode_hash(src)
        return self._sources.get(node_hash, None)


@final
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
        self.sep_layers = layer_separator if layer_separator is not None else SchemaTrackingContextManager._default_sep_layers
        self.sep_structs = struct_separator if struct_separator is not None else SchemaTrackingContextManager._default_sep_structs
        self._source_handler = SchemaSourceHandler()
        self.block_start: blocks.Block|None = None
        self.block_end: blocks.Block|None = None
        self._latest_op: dict|Mapping = {}  # latest schema changes
        self._latest_callee: nodes.BigFrameNode|None = None
        self._func: Callable|None = None
        self._op_count = 0

    @property
    def num_nested_commands(self) -> int:
        return self._op_count

    def prev_changes(self) -> tuple[nodes.BigFrameNode, dict|Mapping]:
        return ((self._latest_callee, self._latest_op)) # type: ignore

    def add_changes(self, hdl: nodes.BigFrameNode, changes: dict|Mapping, fct: Callable|None=None):
        self._latest_callee = hdl
        self._latest_op = changes
        self._func = fct
        self._op_count += 1

    def latest_changes(self) -> list:
        return [self._latest_callee, self._latest_op, self._op_count]

    #@property
    @classmethod
    def active(cls):
        """
        Returns True if context manager is active, ie if we are within a "with" block
        """
        return cls._is_active

    def get_source(self, src: int) -> DiGraph:
        return self._source_handler.sources.get(src, None)     

    def reset_block_markers(self):
        self.block_start = None
        self.block_end = None
        return

    # Context Manager interface
    def __enter__(self):
        assert(config.options.bigquery.project is not None and config.options.bigquery.location is not None)
        SchemaTrackingContextManager._is_active = True
        self.reset_block_markers()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        #self._dag_to_schema()
        SchemaTrackingContextManager._is_active = False
        #TODO: compute final schema from DAG
        #TODO: delete DAG so "new" context can be used
        #TODO: Get new source in case of joins. new table name/ which one is target?
        return

    # Private helper methods for starting schema deduction and DAG creation
    @staticmethod
    def _has_nested_data(schema: list) -> bool:
        return sum([1 for x in schema if x.field_type == "RECORD"]) > 0

    def explode_nested(self, df: DataFrame, sep_explode: str, sep_struct: str|None=None, columns: list|None=None) -> tuple[DataFrame, dict[str, pa_datatype]]:
        """
        :param bigframes.dataframe.DataFrame df: DataFrame to explode
        :param str sep_explode: separator used in exploded representation
        :param str sep_struct: separator used in BigQuery for separating structs. Default: "."
        :param list[str] colums: columns to explode, using sep_struct as a separator
        :returns tuple[bigframes.dataframe.DataFrame, dict[str, str]]: Returns exploded data frame
            and its schema, layers separated by sep_explode

        The methods explodes a potentially nested DataFrame in a BFS like manner:
        We traverse all columns and explode whenever we find a nested/struct like one.
        If one is found and exploded, we restart. This way we can explode all layers without having to select sub-frames,
        iow no depth processing is done at all.
        """
        sep_struct = sep_struct if sep_struct is not None else "."
        schema_ret = {}
        df_flattened = df.copy()
        prefixes = []

        nested_col = [""]
        while nested_col:
            schema = df_flattened.dtypes.to_dict()
            assert(isinstance(schema, dict))
            nested_col = []
            parent = ""
            for col, dtp in schema.items(): 
                pref = col.rsplit(sep_explode, 1)[0]
                _parent = [p for p in prefixes if pref == p.rsplit(sep_explode, 1)[0]]
                _parent = _parent[0].rstrip(sep_explode) if len(_parent) > 0 else parent
                if dt_is_struct_like(dtp):
                    nested_col.append(col)
                    prefixes.append(col+sep_explode)
                value = tuple((_parent, dtp))
                if schema_ret.get(col, None) is None:
                    schema_ret[col] = value
                #TODO: re-insert selecting columns when working
                #cols_considered = nested_col if columns is None else columns
                #nested_col = [cc for cc in cols_considered if cc.startswith(tuple(nested_col))]
                if nested_col:
                    continue  # restart after having exploded
            if nested_col:
                df_flattened = df_flattened.struct.explode(nested_col[0], separator=sep_explode)

        # finalize adding non nested columns to schema
        for col, dtp in schema.items():
            if schema_ret.get(col, None) is None:
                schema_ret[col] = tuple(("", dtp))
            
        return tuple((df_flattened, schema_ret)) # type: ignore

    def add(self, df: DataFrame, layer_separator: str, struct_separator: str):
        df_flattened, df_schema = self.explode_nested(df, sep_explode=layer_separator, sep_struct=struct_separator)
        schema = df_flattened.dtypes
        cols = list(df_flattened.dtypes.keys())
        node = df_flattened.block.expr.node
        dag = self._source_handler.dag_from_df(schema=df_schema, layer_separator=layer_separator, struct_separator=struct_separator)
        schema_orig: Tuple[SchemaField, ...] = node.physical_schema if hasattr(node, "physical_schema") else None # type: ignore
        source = SchemaSource(node_flattened=node, dag=dag, schema=schema, schema_orig=schema_orig) # type: ignore  # noqa: E999
        hash_df = bfnode_hash(df.block.expr.node)
        self._source_handler.add_source(hash_df=hash_df, source = source)

    def lineage(self, df) -> dict:
        node = df.block.expr.node
        return self._source_handler.sources.get(node, {})
       
# Work In Progress

    def get_cols_changes(self):
        cols = list(self._latest_op.keys())
        cols = [col.replace(self.sep_structs, self.sep_layers) for col in cols]
        dag = self._source_handler.sources[self._latest_callee]
        pass

    def step(self):  #nodes.BigFrameNode):
        assert(self.block_start is not None)
        assert(self.block_end is not None)
        hash_start = bfnode_hash(self.block_start.expr.node)
        hash_parent = bfnode_hash(self.block_end.expr.node.child)
        assert(hash_start==hash_parent)
        hdl = None
        if hash_start:
            hdl = self._source_handler.sources.get(hash_parent, None)
            if hdl is None:
                raise ValueError(f"NestedDataCM: Unknown data source {self.block_start}")
            
        # no join, merge etc., no new source/BigFrameNode
        else:
            if not self._schemata_matching(self.block_start, hdl):
                raise Exception("Internal error: Nested Schema mismatch")
                self._extend_dag(hdl, self.block_end.expr.node)

# Default arguments are only evaluated once when the function is created
def schema_tracking_factory(singleton=SchemaTrackingContextManager()) -> SchemaTrackingContextManager:
    return singleton
