#todo: copyright cop pas
# base: sql templates
# use recursive schema for creating exploded column names, sparated by dots. howver, maybe join
# called by part of series.apply, function applying cellwise to rows

# use read_gqb on data coming from Tim

from google.cloud.bigquery.schema import SchemaField
from google.cloud.bigquery_storage_v1 import types
from dataclasses import dataclass
from collections import namedtuple
import pyarrow
from abc import ABCMeta, abstractmethod, ABC
from typing import List, Iterable
from copy import deepcopy
from bigframes.functions.nested_utils import MemberSelector


TableDescription = namedtuple("TableDescription", ["parent", "project", "dataset", "table", "table_fq", "table_full"])


class TFDataFormat(MemberSelector):
    T_ARROW = types.DataFormat.ARROW
    T_AVRO = types.DataFormat.AVRO

        
class BQSchemaLayout(MemberSelector):
    # fm = (f)ield(m)ode
    T_fm_record: str = "RECORD"
    T_fm_repeated: str = "REPEATED"
    T_fm_nullable: str = "NULLABLE"
    T_fm_required: str = "REQUIRED"
    T_field: str = "field"
    T_name: str = "name"
    T_mode: str = "mode"
    T_output_type: str = "output_type"
    T_fields: str = "fields"
    T_field_type: str = "field_type"
    T_type: str = "type"
    
    def __init__(self, schema: List[SchemaField], data_format: TFDataFormat | None = None):
        """
        :param Union[List[SchemaField], DSchemaTable] schema: The schema, result of calling BQTools.table_schema, of a BQ table
        """
        self.bq_schema = schema
        self.data_format = data_format if data_format is not None else TFDataFormat.T_ARROW
        # these need to be reset:
        self._visited = {}
        self._map_to_list = {}
        self._map_to_type = {}

    def _reset(self) -> None:
        self._visited = {}
        self._map_to_list = {}
        self._map_to_type = {}
        return
                
    def _unroll_schema(self, current_field: SchemaField, current_hierarchy: List[str],
                       sep: str) -> None:
        field_name = current_field.name
        fields = current_field.fields
        if not self._visited.get(field_name, False):
            hierarchy = current_hierarchy + [field_name]
            col_name_nested = sep.join(hierarchy)
            self._visited[field_name] = True
            self._map_to_type[col_name_nested] = current_field.field_type
            self._map_to_list[col_name_nested] = hierarchy

            if fields: # no record but a primitive -> end of recursion! 
                for field_value in fields:
                    self._unroll_schema(field_value, hierarchy, sep=sep)

    @property
    def map_to_list(self) -> dict:
        return self._map_to_list

    @property
    def map_to_type(self) -> dict:
        return self._map_to_type
    
    def determine_layout(self, struc_separator: str):
        """
        :param result_shape: Determines output shape: just sequence, or stack features up to level x (reverse from deepest level of nesting):
        :param stack_level:
        :param data_format:
        :return:
        """
        self._reset()
        _schema = [entry for idx, entry in enumerate(self.bq_schema)]
        current_hierarchy = []
        for field_idx, field_value in enumerate(_schema):
            self._unroll_schema(field_value, current_hierarchy, sep=struc_separator)
        return
