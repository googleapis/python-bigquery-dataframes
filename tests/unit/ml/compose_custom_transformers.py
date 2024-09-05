import bigframes.pandas as bpd
from bigframes.ml.compose import CustomTransformer
from typing import List, Optional, Union, Dict
import re


class IdentityTransformer(CustomTransformer):
    _CTID = "IDENT"
    IDENT_BQSQL_RX = re.compile("^(?P<colname>[a-z][a-z0-9_]+)$", flags=re.IGNORECASE)

    def custom_compile_to_sql(self, X: bpd.DataFrame, column: str) -> str:
        return f"{column}"

    @classmethod
    def custom_parse_from_sql(
        cls, config: Optional[Union[Dict, List]], sql: str
    ) -> tuple[CustomTransformer, str]:
        col_label = cls.IDENT_BQSQL_RX.match(sql).group("colname")
        return cls(), col_label


CustomTransformer.register(IdentityTransformer)


class Length1Transformer(CustomTransformer):
    _CTID = "LEN1"
    _DEFAULT_VALUE_DEFAULT = -1
    LEN1_BQSQL_RX = re.compile(
        "^CASE WHEN (?P<colname>[a-z][a-z0-9_]*) IS NULL THEN (?P<defaultvalue>[-]?[0-9]+) ELSE LENGTH[(](?P=colname)[)] END$",
        flags=re.IGNORECASE,
    )

    def __init__(self, default_value: Optional[int] = None):
        self.default_value = default_value

    def custom_compile_to_sql(self, X: bpd.DataFrame, column: str) -> str:
        default_value = (
            self.default_value
            if self.default_value is not None
            else Length1Transformer._DEFAULT_VALUE_DEFAULT
        )
        return (
            f"CASE WHEN {column} IS NULL THEN {default_value} ELSE LENGTH({column}) END"
        )

    @classmethod
    def custom_parse_from_sql(
        cls, config: Optional[Union[Dict, List]], sql: str
    ) -> tuple[CustomTransformer, str]:
        m = cls.LEN1_BQSQL_RX.match(sql)
        col_label = m.group("colname")
        default_value = int(m.group("defaultvalue"))
        return cls(default_value), col_label


CustomTransformer.register(Length1Transformer)


class Length2Transformer(CustomTransformer):
    _CTID = "LEN2"
    _DEFAULT_VALUE_DEFAULT = -1
    LEN2_BQSQL_RX = re.compile(
        "^CASE WHEN (?P<colname>[a-z][a-z0-9_]*) .*$", flags=re.IGNORECASE
    )

    def __init__(self, default_value: Optional[int] = None):
        self.default_value = default_value

    def get_persistent_config(self, column: str) -> Optional[Union[Dict, List]]:
        return [self.default_value]

    def custom_compile_to_sql(self, X: bpd.DataFrame, column: str) -> str:
        default_value = (
            self.default_value
            if self.default_value is not None
            else Length2Transformer._DEFAULT_VALUE_DEFAULT
        )
        return (
            f"CASE WHEN {column} IS NULL THEN {default_value} ELSE LENGTH({column}) END"
        )

    @classmethod
    def custom_parse_from_sql(
        cls, config: Optional[Union[Dict, List]], sql: str
    ) -> tuple[CustomTransformer, str]:
        col_label = cls.LEN2_BQSQL_RX.match(sql).group("colname")
        default_value = config[0]  # get default value from persistent_config
        return cls(default_value), col_label


CustomTransformer.register(Length2Transformer)
