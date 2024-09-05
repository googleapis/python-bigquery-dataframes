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
import pytest

import sklearn.compose as sklearn_compose  # type: ignore
import sklearn.preprocessing as sklearn_preprocessing  # type: ignore

from bigframes.ml import compose, preprocessing
from bigframes.ml.compose import ColumnTransformer

from tests.unit.ml.compose_custom_transformers import IdentityTransformer, Length1Transformer, Length2Transformer

from google.cloud import bigquery
from unittest import mock
from bigframes.ml.preprocessing import LabelEncoder


def test_columntransformer_init_expectedtransforms():
    onehot_transformer = preprocessing.OneHotEncoder()
    standard_scaler_transformer = preprocessing.StandardScaler()
    max_abs_scaler_transformer = preprocessing.MaxAbsScaler()
    min_max_scaler_transformer = preprocessing.MinMaxScaler()
    k_bins_discretizer_transformer = preprocessing.KBinsDiscretizer(strategy="uniform")
    label_transformer = preprocessing.LabelEncoder()
    column_transformer = compose.ColumnTransformer(
        [
            ("onehot", onehot_transformer, "species"),
            (
                "standard_scale",
                standard_scaler_transformer,
                ["culmen_length_mm", "flipper_length_mm"],
            ),
            (
                "max_abs_scale",
                max_abs_scaler_transformer,
                ["culmen_length_mm", "flipper_length_mm"],
            ),
            (
                "min_max_scale",
                min_max_scaler_transformer,
                ["culmen_length_mm", "flipper_length_mm"],
            ),
            (
                "k_bins_discretizer",
                k_bins_discretizer_transformer,
                ["culmen_length_mm", "flipper_length_mm"],
            ),
            ("label", label_transformer, "species"),
        ]
    )

    assert column_transformer.transformers_ == [
        ("onehot", onehot_transformer, "species"),
        ("standard_scale", standard_scaler_transformer, "culmen_length_mm"),
        ("standard_scale", standard_scaler_transformer, "flipper_length_mm"),
        ("max_abs_scale", max_abs_scaler_transformer, "culmen_length_mm"),
        ("max_abs_scale", max_abs_scaler_transformer, "flipper_length_mm"),
        ("min_max_scale", min_max_scaler_transformer, "culmen_length_mm"),
        ("min_max_scale", min_max_scaler_transformer, "flipper_length_mm"),
        ("k_bins_discretizer", k_bins_discretizer_transformer, "culmen_length_mm"),
        ("k_bins_discretizer", k_bins_discretizer_transformer, "flipper_length_mm"),
        ("label", label_transformer, "species"),
    ]


def test_columntransformer_repr():
    column_transformer = compose.ColumnTransformer(
        [
            (
                "onehot",
                preprocessing.OneHotEncoder(),
                "species",
            ),
            (
                "standard_scale",
                preprocessing.StandardScaler(),
                ["culmen_length_mm", "flipper_length_mm"],
            ),
            (
                "max_abs_scale",
                preprocessing.MaxAbsScaler(),
                ["culmen_length_mm", "flipper_length_mm"],
            ),
            (
                "min_max_scale",
                preprocessing.MinMaxScaler(),
                ["culmen_length_mm", "flipper_length_mm"],
            ),
            (
                "k_bins_discretizer",
                preprocessing.KBinsDiscretizer(strategy="uniform"),
                ["culmen_length_mm", "flipper_length_mm"],
            ),
        ]
    )

    assert (
        column_transformer.__repr__()
        == """ColumnTransformer(transformers=[('onehot', OneHotEncoder(), 'species'),
                                ('standard_scale', StandardScaler(),
                                 ['culmen_length_mm', 'flipper_length_mm']),
                                ('max_abs_scale', MaxAbsScaler(),
                                 ['culmen_length_mm', 'flipper_length_mm']),
                                ('min_max_scale', MinMaxScaler(),
                                 ['culmen_length_mm', 'flipper_length_mm']),
                                ('k_bins_discretizer',
                                 KBinsDiscretizer(strategy='uniform'),
                                 ['culmen_length_mm', 'flipper_length_mm'])])"""
    )


def test_columntransformer_repr_matches_sklearn():
    bf_column_transformer = compose.ColumnTransformer(
        [
            (
                "onehot",
                preprocessing.OneHotEncoder(),
                "species",
            ),
            (
                "standard_scale",
                preprocessing.StandardScaler(),
                ["culmen_length_mm", "flipper_length_mm"],
            ),
            (
                "max_abs_scale",
                preprocessing.MaxAbsScaler(),
                ["culmen_length_mm", "flipper_length_mm"],
            ),
            (
                "min_max_scale",
                preprocessing.MinMaxScaler(),
                ["culmen_length_mm", "flipper_length_mm"],
            ),
            (
                "k_bins_discretizer",
                preprocessing.KBinsDiscretizer(strategy="uniform"),
                ["culmen_length_mm", "flipper_length_mm"],
            ),
        ]
    )
    sk_column_transformer = sklearn_compose.ColumnTransformer(
        [
            (
                "onehot",
                sklearn_preprocessing.OneHotEncoder(),
                "species",
            ),
            (
                "standard_scale",
                sklearn_preprocessing.StandardScaler(),
                ["culmen_length_mm", "flipper_length_mm"],
            ),
            (
                "max_abs_scale",
                sklearn_preprocessing.MaxAbsScaler(),
                ["culmen_length_mm", "flipper_length_mm"],
            ),
            (
                "min_max_scale",
                sklearn_preprocessing.MinMaxScaler(),
                ["culmen_length_mm", "flipper_length_mm"],
            ),
            (
                "k_bins_discretizer",
                sklearn_preprocessing.KBinsDiscretizer(strategy="uniform"),
                ["culmen_length_mm", "flipper_length_mm"],
            ),
        ]
    )

    assert bf_column_transformer.__repr__() == sk_column_transformer.__repr__()


def test_columntransformer_init_with_customtransforms():
    ident_transformer = IdentityTransformer()
    len1_transformer = Length1Transformer(-2)
    len2_transformer = Length2Transformer(99)
    label_transformer = preprocessing.LabelEncoder()
    column_transformer = compose.ColumnTransformer(
        [
            ("ident_trafo", ident_transformer, ["culmen_length_mm", "flipper_length_mm"]),
            ("len1_trafo", len1_transformer, ["species"]),
            ("len2_trafo", len2_transformer, ["species"]),
            ("label", label_transformer, "species"),
        ]
    )

    assert column_transformer.transformers_ == [
        ("ident_trafo", ident_transformer, "culmen_length_mm"),
        ("ident_trafo", ident_transformer, "flipper_length_mm"),
        ("len1_trafo", len1_transformer, "species"),
        ("len2_trafo", len2_transformer, "species"),
        ("label", label_transformer, "species"),
    ]

def test_columntransformer_repr_customtransforms():
    ident_transformer = IdentityTransformer()
    len1_transformer = Length1Transformer(-2)
    len2_transformer = Length2Transformer(99)
    label_transformer = preprocessing.LabelEncoder()
    column_transformer = compose.ColumnTransformer(
        [
            ("ident_trafo", ident_transformer, ["culmen_length_mm", "flipper_length_mm"]),
            ("len1_trafo", len1_transformer, ["species"]),
            ("len2_trafo", len2_transformer, ["species"]),
            ("label", label_transformer, "species")
        ]
    )

    assert (
        column_transformer.__repr__()
        == """ColumnTransformer(transformers=[('ident_trafo', IdentityTransformer(),
                                 ['culmen_length_mm', 'flipper_length_mm']),
                                ('len1_trafo',
                                 Length1Transformer(default_value=-2),
                                 ['species']),
                                ('len2_trafo',
                                 Length2Transformer(default_value=99),
                                 ['species']),
                                ('label', LabelEncoder(), 'species')])"""
    )


IDENT_SQL = "column /*CT.IDENT()*/"
LEN1_SQL = "CASE WHEN column IS NULL THEN -5 ELSE LENGTH(column) END /*CT.LEN1()*/"
LEN2_SQL = "CASE WHEN column IS NULL THEN 99 ELSE LENGTH(column) END /*CT.LEN2([99])*/"
UNKNOWN_CT_SQL = "column /*CT.UNKNOWN()*/"
FOREIGN_SQL = "column"

def test_customtransformer_registry():
    
    compose.CustomTransformer.register(IdentityTransformer)
    compose.CustomTransformer.register(Length1Transformer)
    compose.CustomTransformer.register(Length2Transformer)

    transformer_cls = compose.CustomTransformer.find_matching_transformer(IDENT_SQL)
    assert transformer_cls == IdentityTransformer
    
    transformer_cls = compose.CustomTransformer.find_matching_transformer(LEN1_SQL)
    assert transformer_cls == Length1Transformer
    
    transformer_cls = compose.CustomTransformer.find_matching_transformer(LEN2_SQL)
    assert transformer_cls == Length2Transformer
    
    transformer_cls = compose.CustomTransformer.find_matching_transformer(UNKNOWN_CT_SQL)
    assert transformer_cls == None
    
    transformer_cls = compose.CustomTransformer.find_matching_transformer(FOREIGN_SQL)
    assert transformer_cls == None


def test_customtransformer_compile_sql():
    
    ident_trafo = IdentityTransformer()
    sqls = ident_trafo._compile_to_sql(X=None, columns=["col1", "col2"])
    assert sqls == [
        'col1 /*CT.IDENT()*/ AS ident_col1', 
        'col2 /*CT.IDENT()*/ AS ident_col2'
    ]

    len1_trafo = Length1Transformer(-5)
    sqls = len1_trafo._compile_to_sql(X=None, columns=["col1", "col2"])
    assert sqls == [
        'CASE WHEN col1 IS NULL THEN -5 ELSE LENGTH(col1) END /*CT.LEN1()*/ AS len1_col1', 
        'CASE WHEN col2 IS NULL THEN -5 ELSE LENGTH(col2) END /*CT.LEN1()*/ AS len1_col2'
    ]

    len2_trafo = Length2Transformer(99)
    sqls = len2_trafo._compile_to_sql(X=None, columns=["col1", "col2"])
    assert sqls == [
        'CASE WHEN col1 IS NULL THEN 99 ELSE LENGTH(col1) END /*CT.LEN2([99])*/ AS len2_col1', 
        'CASE WHEN col2 IS NULL THEN 99 ELSE LENGTH(col2) END /*CT.LEN2([99])*/ AS len2_col2'
    ]


def test_customtransformer_parse_sql():
    
    ident_trafo, col_label = IdentityTransformer._parse_from_sql(IDENT_SQL)
    assert col_label == "column"
    assert ident_trafo
    assert isinstance(ident_trafo, IdentityTransformer)

    len1_trafo, col_label = Length1Transformer._parse_from_sql(LEN1_SQL)
    assert col_label == "column"
    assert len1_trafo
    assert isinstance(len1_trafo, Length1Transformer)
    assert len1_trafo.default_value == -5

    len2_trafo, col_label = Length2Transformer._parse_from_sql(LEN2_SQL)
    assert col_label == "column"
    assert len2_trafo
    assert isinstance(len2_trafo, Length2Transformer)
    assert len2_trafo.default_value == 99

    fake_len2_sql = LEN2_SQL.replace("/*CT.LEN2([99])*/", "/*CT.LEN2([77])*/") 
    len2_trafo, col_label = Length2Transformer._parse_from_sql(fake_len2_sql)
    assert col_label == "column"
    assert len2_trafo
    assert isinstance(len2_trafo, Length2Transformer)
    assert len2_trafo.default_value == 77




def create_bq_model_mock(transform_columns, feature_columns=None):
    class _NameClass:
        def __init__(self, name):
            self.name = name
    properties = {"transformColumns": transform_columns}
    mock_bq_model = bigquery.Model("model_project.model_dataset.model_id")
    type(mock_bq_model)._properties = mock.PropertyMock(return_value=properties)
    if feature_columns:
        type(mock_bq_model).feature_columns = mock.PropertyMock(return_value=[_NameClass(col) for col in feature_columns])
    return mock_bq_model


@pytest.fixture
def bq_model_good():
    return create_bq_model_mock([
            {'name': 'ident_culmen_length_mm', 'type': {'typeKind': 'INT64'}, 'transformSql': 'culmen_length_mm /*CT.IDENT()*/'}, 
            {'name': 'ident_flipper_length_mm', 'type': {'typeKind': 'INT64'}, 'transformSql': 'flipper_length_mm /*CT.IDENT()*/'}, 
            {'name': 'len1_species', 'type': {'typeKind': 'INT64'}, 'transformSql': 'CASE WHEN species IS NULL THEN -5 ELSE LENGTH(species) END /*CT.LEN1()*/'}, 
            {'name': 'len2_species', 'type': {'typeKind': 'INT64'}, 'transformSql': 'CASE WHEN species IS NULL THEN 99 ELSE LENGTH(address) END /*CT.LEN2([99])*/'}, 
            {'name': 'labelencoded_county', 'type': {'typeKind': 'INT64'}, 'transformSql': 'ML.LABEL_ENCODER(county, 1000000, 0) OVER()'}, 
            {'name': 'labelencoded_species', 'type': {'typeKind': 'INT64'}, 'transformSql': 'ML.LABEL_ENCODER(species, 1000000, 0) OVER()'}
        ])

@pytest.fixture
def bq_model_merge():
    return create_bq_model_mock([
            {'name': 'labelencoded_county', 'type': {'typeKind': 'INT64'}, 'transformSql': 'ML.LABEL_ENCODER(county, 1000000, 0) OVER()'}, 
            {'name': 'labelencoded_species', 'type': {'typeKind': 'INT64'}, 'transformSql': 'ML.LABEL_ENCODER(species, 1000000, 0) OVER()'}
        ], 
        ["county", "species"])

@pytest.fixture
def bq_model_no_merge():
    return create_bq_model_mock([
            {'name': 'ident_culmen_length_mm', 'type': {'typeKind': 'INT64'}, 'transformSql': 'culmen_length_mm /*CT.IDENT()*/'}
        ],
        ["culmen_length_mm"])

@pytest.fixture
def bq_model_unknown_CT():
    return create_bq_model_mock([
            {'name': 'unknownct_culmen_length_mm', 'type': {'typeKind': 'INT64'}, 'transformSql': 'culmen_length_mm /*CT.UNKNOWN()*/'},
            {'name': 'labelencoded_county', 'type': {'typeKind': 'INT64'}, 'transformSql': 'ML.LABEL_ENCODER(county, 1000000, 0) OVER()'}, 
        ])

@pytest.fixture
def bq_model_unknown_ML():
    return create_bq_model_mock([
            {'name': 'unknownml_culmen_length_mm', 'type': {'typeKind': 'INT64'}, 'transformSql': 'ML.UNKNOWN(culmen_length_mm)'},
            {'name': 'labelencoded_county', 'type': {'typeKind': 'INT64'}, 'transformSql': 'ML.LABEL_ENCODER(county, 1000000, 0) OVER()'}, 
        ])

@pytest.fixture
def bq_model_foreign():
    return create_bq_model_mock([
            {'name': 'foreign_culmen_length_mm', 'type': {'typeKind': 'INT64'}, 'transformSql': 'culmen_length_mm+1'},
            {'name': 'labelencoded_county', 'type': {'typeKind': 'INT64'}, 'transformSql': 'ML.LABEL_ENCODER(county, 1000000, 0) OVER()'}, 
        ])


def test_columntransformer_extract_from_bq_model_good(bq_model_good):
    col_trans = ColumnTransformer._extract_from_bq_model(bq_model_good)
    assert len(col_trans.transformers) == 6
    # normalize the representation for string comparing 
    col_trans.transformers.sort()
    actual = col_trans.__repr__()
    expected = """ColumnTransformer(transformers=[('identity_transformer', IdentityTransformer(),
                                 'culmen_length_mm'),
                                ('identity_transformer', IdentityTransformer(),
                                 'flipper_length_mm'),
                                ('label_encoder',
                                 LabelEncoder(max_categories=1000001,
                                              min_frequency=0),
                                 'county'),
                                ('label_encoder',
                                 LabelEncoder(max_categories=1000001,
                                              min_frequency=0),
                                 'species'),
                                ('length1_transformer',
                                 Length1Transformer(default_value=-5),
                                 'species'),
                                ('length2_transformer',
                                 Length2Transformer(default_value=99),
                                 'species')])"""
    assert expected == actual


def test_columntransformer_extract_from_bq_model_merge(bq_model_merge):
    col_trans = ColumnTransformer._extract_from_bq_model(bq_model_merge)
    assert isinstance(col_trans, ColumnTransformer)
    col_trans = col_trans._merge(bq_model_merge)
    assert isinstance(col_trans, LabelEncoder)
    assert col_trans.__repr__() == """LabelEncoder(max_categories=1000001, min_frequency=0)"""
    assert col_trans._output_names == ['labelencoded_county', 'labelencoded_species']
    
def test_columntransformer_extract_from_bq_model_no_merge(bq_model_no_merge):
    col_trans = ColumnTransformer._extract_from_bq_model(bq_model_no_merge)
    col_trans = col_trans._merge(bq_model_no_merge)
    assert isinstance(col_trans, ColumnTransformer)
    assert col_trans.__repr__() == """ColumnTransformer(transformers=[('identity_transformer', IdentityTransformer(),
                                 'culmen_length_mm')])"""
    
def test_columntransformer_extract_from_bq_model_unknown_CT(bq_model_unknown_CT):
    try:
        col_trans = ColumnTransformer._extract_from_bq_model(bq_model_unknown_CT)
        assert False
    except ValueError as e:
        assert "Missing custom transformer" == e.args[0]

def test_columntransformer_extract_from_bq_model_unknown_ML(bq_model_unknown_ML):
    try:
        col_trans = ColumnTransformer._extract_from_bq_model(bq_model_unknown_ML)
        assert False
    except NotImplementedError as e:
        assert "Unsupported transformer type" in e.args[0]

def test_columntransformer_extract_from_bq_model_foreign(bq_model_foreign):
    col_trans = ColumnTransformer._extract_from_bq_model(bq_model_foreign)
    assert col_trans.__repr__() == """ColumnTransformer(transformers=[('label_encoder',
                                 LabelEncoder(max_categories=1000001,
                                              min_frequency=0),
                                 'county')])"""


def test_columntransformer_extract_output_names(bq_model_good):
    class BQMLModel:
        def __init__(self, bq_model):
            self._model = bq_model
    col_trans = ColumnTransformer._extract_from_bq_model(bq_model_good)
    col_trans._bqml_model = BQMLModel(bq_model_good)
    col_trans._extract_output_names()
    assert col_trans._output_names == [
        'ident_culmen_length_mm', 
        'ident_flipper_length_mm', 
        'len1_species', 
        'len2_species', 
        'labelencoded_county', 
        'labelencoded_species'
    ]
    

def test_columntransformer_compile_to_sql():    
    ident_transformer = IdentityTransformer()
    len1_transformer = Length1Transformer(-2)
    len2_transformer = Length2Transformer(99)
    label_transformer = preprocessing.LabelEncoder()
    column_transformer = compose.ColumnTransformer(
        [
            ("ident_trafo", ident_transformer, ["culmen_length_mm", "flipper_length_mm"]),
            ("len1_trafo", len1_transformer, ["species"]),
            ("len2_trafo", len2_transformer, ["species"]),
            ("label", label_transformer, "species"),
        ]
    )
    sqls = column_transformer._compile_to_sql(None)
    assert sqls == [
        'culmen_length_mm /*CT.IDENT()*/ AS ident_culmen_length_mm', 
        'flipper_length_mm /*CT.IDENT()*/ AS ident_flipper_length_mm', 
        'CASE WHEN species IS NULL THEN -2 ELSE LENGTH(species) END /*CT.LEN1()*/ AS len1_species', 
        'CASE WHEN species IS NULL THEN 99 ELSE LENGTH(species) END /*CT.LEN2([99])*/ AS len2_species', 
        'ML.LABEL_ENCODER(species, 1000000, 0) OVER() AS labelencoded_species'
    ]
