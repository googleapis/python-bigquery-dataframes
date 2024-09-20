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

"""Build composite transformers on heterogeneous data. This module is styled
after scikit-Learn's compose module:
https://scikit-learn.org/stable/modules/classes.html#module-sklearn.compose."""

from __future__ import annotations

import re
import types
import typing
from typing import cast, Iterable, List, Optional, Set, Tuple, Union, Dict, Type
import abc
import json

from bigframes_vendored import constants
import bigframes_vendored.sklearn.compose._column_transformer
from google.cloud import bigquery

from bigframes.core import log_adapter
from bigframes.ml import base, core, globals, impute, preprocessing, utils
import bigframes.pandas as bpd

_BQML_TRANSFROM_TYPE_MAPPING = types.MappingProxyType(
    {
        "ML.STANDARD_SCALER": preprocessing.StandardScaler,
        "ML.ONE_HOT_ENCODER": preprocessing.OneHotEncoder,
        "ML.MAX_ABS_SCALER": preprocessing.MaxAbsScaler,
        "ML.MIN_MAX_SCALER": preprocessing.MinMaxScaler,
        "ML.BUCKETIZE": preprocessing.KBinsDiscretizer,
        "ML.QUANTILE_BUCKETIZE": preprocessing.KBinsDiscretizer,
        "ML.LABEL_ENCODER": preprocessing.LabelEncoder,
        "ML.POLYNOMIAL_EXPAND": preprocessing.PolynomialFeatures,
        "ML.IMPUTER": impute.SimpleImputer,
    }
)


CUSTOM_TRANSFORMER_SQL_RX = re.compile(
    "^(?P<sql>.*)/[*]CT.(?P<id>[A-Z]+[A-Z0-9]*)[(](?P<config>[^*]*)[)][*]/$",
    re.IGNORECASE,
)


class SQLScalarColumnTransformer(base.BaseTransformer):
    def __init__(self, sql: str, target_column="transformed_{0}"):
        super().__init__()
        self.sql = sql
        self.target_column = target_column

    def _compile_to_sql(
        self, X: bpd.DataFrame, columns: Optional[Iterable[str]] = None
    ) -> List[str]:
        if columns is None:
            columns = X.columns
        result = []
        for column in columns:
            current_sql = self.sql.format(column)
            current_target_column = self.target_column.format(column)
            result.append(f"{current_sql} AS {current_target_column}")
        return result

    def _keys(self):
        return (self.sql, self.target_column)

    def __lt__(self, other):
        return self.target_column < other.target_column


class CustomTransformer(base.BaseTransformer):
    _CTID = None
    _custom_transformer_classes = {}

    @classmethod
    def register(cls, transformer_cls: Type[base.BaseTransformer]):
        assert transformer_cls._CTID
        cls._custom_transformer_classes[transformer_cls._CTID] = transformer_cls

    @classmethod
    def find_matching_transformer(
        cls, transform_sql: str
    ) -> Optional[Type[base.BaseTransformer]]:
        for transform_cls in cls._custom_transformer_classes.values():
            if transform_cls.understands(transform_sql):
                return transform_cls
        return None

    @classmethod
    def understands(cls, transform_sql: str) -> bool:
        """
        may be overwritten to have a more advanced matching, possibly without comments in SQL
        """
        m = CUSTOM_TRANSFORMER_SQL_RX.match(transform_sql)
        if m and m.group("id").strip() == cls._CTID:
            return True
        return False

    def __init__(self):
        super().__init__()

    def _compile_to_sql(
        self, X: bpd.DataFrame, columns: Optional[Iterable[str]] = None
    ) -> List[str]:
        if columns is None:
            columns = X.columns
        return [
            f"{self.custom_compile_to_sql(X, column)} {self._get_sql_comment(column)} AS {self.get_target_column_name(column)}"
            for column in columns
        ]

    def get_target_column_name(self, column: str) -> str:
        return f"{self._CTID.lower()}_{column}"

    @classmethod
    @abc.abstractclassmethod
    def custom_compile_to_sql(cls, X: bpd.DataFrame, column: str) -> str:
        pass

    def get_persistent_config(self, column: str) -> Optional[Union[Dict, List]]:
        """
        return structure to be persisted in the comment of the sql
        """
        return None

    def _get_pc_as_args(self, column: str) -> str:
        config = self.get_persistent_config(column)
        if not config:
            return ""
        return json.dumps(config)

    def _get_sql_comment(self, column: str) -> str:
        args = self._get_pc_as_args(column)
        return f"/*CT.{self._CTID}({args})*/"

    @classmethod
    def _parse_from_sql(cls, transform_sql: str) -> Tuple[base.BaseTransformer, str]:
        m = CUSTOM_TRANSFORMER_SQL_RX.match(transform_sql)
        if m and m.group("id").strip() != cls._CTID:
            raise ValueError("understand() does not match _parse_from_sql!")
        args = m.group("config").strip()
        if args != "":
            config = json.loads(args)
        else:
            config = None
        sql = m.group("sql").strip()
        return cls.custom_parse_from_sql(config, sql)

    @classmethod
    @abc.abstractclassmethod
    def custom_parse_from_sql(
        cls, config: Optional[Union[Dict, List]], sql: str
    ) -> Tuple[base.BaseTransformer, str]:
        """
        return transformer instance and column name
        """
        pass

    def _keys(self):
        return ()

    # CustomTransformers are thought to be used inside a column transformer.
    # So there is no need to implement fit() and transform() directly.
    # ColumnTransformer.merge() takes care, that a single custom transformer
    # is not returned as a standalone transformer.
    def fit(self, y: Union[bpd.DataFrame, bpd.Series]) -> base.BaseTransformer:
        raise NotImplementedError("Unsupported")

    def transform(self, y: Union[bpd.DataFrame, bpd.Series]) -> bpd.DataFrame:
        raise NotImplementedError("Unsupported")


@log_adapter.class_logger
class ColumnTransformer(
    base.Transformer,
    bigframes_vendored.sklearn.compose._column_transformer.ColumnTransformer,
):
    __doc__ = (
        bigframes_vendored.sklearn.compose._column_transformer.ColumnTransformer.__doc__
    )

    def __init__(
        self,
        transformers: Iterable[
            Tuple[
                str,
                Union[preprocessing.PreprocessingType, impute.SimpleImputer],
                Union[str, Iterable[str]],
            ]
        ],
    ):
        # TODO: if any(transformers) has fitted raise warning
        self.transformers = list(transformers)
        self._bqml_model: Optional[core.BqmlModel] = None
        self._bqml_model_factory = globals.bqml_model_factory()
        # call self.transformers_ to check chained transformers
        self.transformers_

    def _keys(self):
        return (self.transformers, self._bqml_model)

    @property
    def transformers_(
        self,
    ) -> List[
        Tuple[str, Union[preprocessing.PreprocessingType, impute.SimpleImputer], str]
    ]:
        """The collection of transformers as tuples of (name, transformer, column)."""
        result: List[
            Tuple[
                str,
                Union[preprocessing.PreprocessingType, impute.SimpleImputer],
                str,
            ]
        ] = []

        for entry in self.transformers:
            name, transformer, column_or_columns = entry
            columns = (
                column_or_columns
                if isinstance(column_or_columns, List)
                else [column_or_columns]
            )

            for column in columns:
                result.append((name, transformer, column))

        return result

    @classmethod
    def _extract_from_bq_model(
        cls,
        bq_model: bigquery.Model,
    ) -> ColumnTransformer:
        """Extract transformers as ColumnTransformer obj from a BQ Model. Keep the _bqml_model field as None."""
        assert "transformColumns" in bq_model._properties

        transformers_set: Set[
            Tuple[
                str,
                Union[preprocessing.PreprocessingType, impute.SimpleImputer],
                Union[str, List[str]],
            ]
        ] = set()

        def camel_to_snake(name):
            name = re.sub("(.)([A-Z][a-z]+)", r"\1_\2", name)
            return re.sub("([a-z0-9])([A-Z])", r"\1_\2", name).lower()

        output_names = []
        for transform_col in bq_model._properties["transformColumns"]:
            transform_col_dict = cast(dict, transform_col)
            # pass the columns that are not transformed
            if "transformSql" not in transform_col_dict:
                continue
            transform_sql: str = transform_col_dict["transformSql"]

            found_transformer = False
            for prefix in _BQML_TRANSFROM_TYPE_MAPPING:
                if transform_sql.startswith(prefix):
                    transformer_cls = _BQML_TRANSFROM_TYPE_MAPPING[prefix]
                    transformers_set.add(
                        (
                            camel_to_snake(transformer_cls.__name__),
                            *transformer_cls._parse_from_sql(transform_sql),  # type: ignore
                        )
                    )

                    found_transformer = True
                    break

            if not found_transformer:
                if transform_sql.startswith("ML."):
                    raise NotImplementedError(
                        f"Unsupported transformer type. {constants.FEEDBACK_LINK}"
                    )

                target_column = transform_col_dict["name"]
                transformer = SQLScalarColumnTransformer(
                    transform_sql, target_column=target_column
                )
                input_column_name = "?"
                transformers_set.add(
                    (
                        camel_to_snake(transformer.__class__.__name__),
                        transformer,
                        input_column_name,
                    )
                )
                found_transformer = True

            output_names.append(transform_col_dict["name"])

        transformer = cls(transformers=list(transformers_set))
        transformer._output_names = output_names

        return transformer

    def _merge(
        self, bq_model: bigquery.Model
    ) -> Union[
        ColumnTransformer, Union[preprocessing.PreprocessingType, impute.SimpleImputer]
    ]:
        """Try to merge the column transformer to a simple transformer. Depends on all the columns in bq_model are transformed with the same transformer."""
        transformers = self.transformers

        assert len(transformers) > 0
        _, transformer_0, column_0 = transformers[0]
        if isinstance(transformer_0, SQLScalarColumnTransformer):
            return self  # SQLScalarColumnTransformer only work inside ColumnTransformer
        feature_columns_sorted = sorted(
            [
                cast(str, feature_column.name)
                for feature_column in bq_model.feature_columns
            ]
        )

        if (
            len(transformers) == 1
            and isinstance(transformer_0, preprocessing.PolynomialFeatures)
            and sorted(column_0) == feature_columns_sorted
        ):
            transformer_0._output_names = self._output_names
            return transformer_0

        if not isinstance(column_0, str):
            return self
        columns = [column_0]
        for _, transformer, column in transformers[1:]:
            if not isinstance(column, str):
                return self
            # all transformers are the same
            if transformer != transformer_0:
                return self
            columns.append(column)
        # all feature columns are transformed
        if sorted(columns) == feature_columns_sorted:
            transformer_0._output_names = self._output_names
            return transformer_0

        return self

    def _compile_to_sql(
        self,
        X: bpd.DataFrame,
    ) -> List[str]:
        """Compile this transformer to a list of SQL expressions that can be included in
        a BQML TRANSFORM clause

        Args:
            X: DataFrame to transform.

        Returns: a list of sql_expr."""
        result = []
        for _, transformer, target_columns in self.transformers:
            if isinstance(target_columns, str):
                target_columns = [target_columns]
            result += transformer._compile_to_sql(X, target_columns)
        return result

    def fit(
        self,
        X: Union[bpd.DataFrame, bpd.Series],
        y=None,  # ignored
    ) -> ColumnTransformer:
        (X,) = utils.convert_to_dataframe(X)

        transform_sqls = self._compile_to_sql(X)
        self._bqml_model = self._bqml_model_factory.create_model(
            X,
            options={"model_type": "transform_only"},
            transforms=transform_sqls,
        )

        self._extract_output_names()
        return self

    # Overwrite the implementation in BaseTransformer, as it only supports the "ML." transformers.
    def _extract_output_names(self):
        """Extract transform output column names. Save the results to self._output_names."""
        assert self._bqml_model is not None

        output_names = []
        for transform_col in self._bqml_model._model._properties["transformColumns"]:
            transform_col_dict = cast(dict, transform_col)
            # pass the columns that are not transformed
            if "transformSql" not in transform_col_dict:
                continue
            transform_sql: str = transform_col_dict["transformSql"]
            if not transform_sql.startswith("ML."):
                if not CustomTransformer.find_matching_transformer(transform_sql):
                    continue

            output_names.append(transform_col_dict["name"])

        self._output_names = output_names

    def transform(self, X: Union[bpd.DataFrame, bpd.Series]) -> bpd.DataFrame:
        if not self._bqml_model:
            raise RuntimeError("Must be fitted before transform")

        (X,) = utils.convert_to_dataframe(X)

        df = self._bqml_model.transform(X)
        return typing.cast(
            bpd.DataFrame,
            df[self._output_names],
        )
