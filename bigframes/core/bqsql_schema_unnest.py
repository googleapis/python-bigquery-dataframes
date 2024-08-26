from google.cloud.bigquery.schema import SchemaField
from google.cloud.bigquery_storage_v1 import types

from bigframes.functions.nested_utils import MemberSelector


# We introduce using MemberSelector to avoid having to remember hard coded strings and make things IDE-able.

class BQSchemaLayout(MemberSelector):
    """
    Unpacks a BigQuery schema into a nested structure.
    """
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

    def __init__(self, schema: list[SchemaField], data_format: types.DataFormat|None = None):
        """
        :param Union[List[SchemaField], DSchemaTable] schema: The schema, result of calling BQTools.table_schema, of a BQ table
        """
        self.bq_schema = schema
        self.data_format = data_format if data_format is not None else types.DataFormat.ARROW
        # these need to be reset:
        self._visited = {}
        self._map_to_list = {}
        self._map_to_type = {}

    @property
    def map_to_list(self) -> dict:
        return self._map_to_list

    @property
    def map_to_type(self) -> dict:
        return self._map_to_type

    def _reset(self) -> None:
        self._visited = {}
        self._map_to_list = {}
        self._map_to_type = {}
        return

    def _unroll_schema(self, current_field: SchemaField, current_hierarchy: list[str], sep: str) -> None:
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

    def determine_layout(self, struc_separator: str):
        """
        :param result_shape: Determines output shape: just sequence, or stack features up to level x (reverse from deepest level of nesting):
        :param stack_level:
        :param data_format:
        :return:
        """
        self._reset()
        schema = [entry for idx, entry in enumerate(self.bq_schema)]
        current_hierarchy = []
        for _, field_value in enumerate(schema):
            self._unroll_schema(field_value, current_hierarchy, sep=struc_separator)
        return
