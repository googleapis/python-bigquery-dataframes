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

from __future__ import annotations

from bigframes.operations.array_ops import ArrayIndexOp, ArraySliceOp, ArrayToStringOp
from bigframes.operations.base_ops import (
    BinaryOp,
    NaryOp,
    RowOp,
    ScalarOp,
    TernaryOp,
    UnaryOp,
)
from bigframes.operations.blob_ops import (
    obj_fetch_metadata_op,
    obj_make_ref_op,
    ObjGetAccessUrl,
)
from bigframes.operations.bool_ops import and_op, or_op, xor_op
from bigframes.operations.comparison_ops import (
    eq_null_match_op,
    eq_op,
    ge_op,
    gt_op,
    le_op,
    lt_op,
    ne_op,
)
from bigframes.operations.date_ops import (
    day_op,
    dayofweek_op,
    month_op,
    quarter_op,
    year_op,
)
from bigframes.operations.datetime_ops import (
    date_op,
    StrftimeOp,
    time_op,
    ToDatetimeOp,
    ToTimestampOp,
)
from bigframes.operations.distance_ops import (
    cosine_distance_op,
    euclidean_distance_op,
    manhattan_distance_op,
)
from bigframes.operations.frequency_ops import (
    DatetimeToIntegerLabelOp,
    FloorDtOp,
    IntegerLabelToDatetimeOp,
)
from bigframes.operations.generic_ops import (
    AsTypeOp,
    case_when_op,
    CaseWhenOp,
    clip_op,
    coalesce_op,
    fillna_op,
    hash_op,
    invert_op,
    IsInOp,
    isnull_op,
    MapOp,
    maximum_op,
    minimum_op,
    notnull_op,
    RowKey,
    where_op,
)
from bigframes.operations.geo_ops import geo_x_op, geo_y_op
from bigframes.operations.json_ops import (
    JSONExtract,
    JSONExtractArray,
    JSONExtractStringArray,
    JSONSet,
    ParseJSON,
    ToJSONString,
)
from bigframes.operations.numeric_ops import (
    abs_op,
    add_op,
    arccos_op,
    arccosh_op,
    arcsin_op,
    arcsinh_op,
    arctan2_op,
    arctan_op,
    arctanh_op,
    ceil_op,
    cos_op,
    cosh_op,
    div_op,
    exp_op,
    expm1_op,
    floor_op,
    floordiv_op,
    ln_op,
    log1p_op,
    log10_op,
    mod_op,
    mul_op,
    neg_op,
    pos_op,
    pow_op,
    round_op,
    sin_op,
    sinh_op,
    sqrt_op,
    sub_op,
    tan_op,
    tanh_op,
    unsafe_pow_op,
)
from bigframes.operations.numpy_op_maps import NUMPY_TO_BINOP, NUMPY_TO_OP
from bigframes.operations.remote_function_ops import (
    BinaryRemoteFunctionOp,
    NaryRemoteFunctionOp,
    RemoteFunctionOp,
)
from bigframes.operations.string_ops import (
    capitalize_op,
    EndsWithOp,
    isalnum_op,
    isalpha_op,
    isdecimal_op,
    isdigit_op,
    islower_op,
    isnumeric_op,
    isspace_op,
    isupper_op,
    len_op,
    lower_op,
    lstrip_op,
    RegexReplaceStrOp,
    ReplaceStrOp,
    reverse_op,
    rstrip_op,
    StartsWithOp,
    strconcat_op,
    StrContainsOp,
    StrContainsRegexOp,
    StrExtractOp,
    StrFindOp,
    StrGetOp,
    StringSplitOp,
    strip_op,
    StrPadOp,
    StrRepeatOp,
    StrSliceOp,
    upper_op,
    ZfillOp,
)
from bigframes.operations.struct_ops import StructFieldOp, StructOp
from bigframes.operations.time_ops import hour_op, minute_op, normalize_op, second_op

__all__ = [
    # Base ops
    "RowOp",
    "NaryOp",
    "UnaryOp",
    "BinaryOp",
    "TernaryOp",
    "ScalarOp",
    # Generic ops
    "AsTypeOp",
    "case_when_op",
    "CaseWhenOp",
    "clip_op",
    "coalesce_op",
    "fillna_op",
    "hash_op",
    "invert_op",
    "IsInOp",
    "isnull_op",
    "MapOp",
    "maximum_op",
    "minimum_op",
    "notnull_op",
    "RowKey",
    "where_op",
    # String ops
    "capitalize_op",
    "EndsWithOp",
    "isalnum_op",
    "isalpha_op",
    "isdecimal_op",
    "isdigit_op",
    "islower_op",
    "isnumeric_op",
    "isspace_op",
    "isupper_op",
    "len_op",
    "lower_op",
    "lstrip_op",
    "RegexReplaceStrOp",
    "ReplaceStrOp",
    "reverse_op",
    "rstrip_op",
    "StartsWithOp",
    "strconcat_op",
    "StrContainsOp",
    "StrContainsRegexOp",
    "StrExtractOp",
    "StrFindOp",
    "StrGetOp",
    "StringSplitOp",
    "strip_op",
    "StrPadOp",
    "StrRepeatOp",
    "StrSliceOp",
    "upper_op",
    "ZfillOp",
    # Date ops
    "day_op",
    "month_op",
    "year_op",
    "dayofweek_op",
    "quarter_op",
    # Time ops
    "hour_op",
    "minute_op",
    "second_op",
    "normalize_op",
    # Datetime ops
    "date_op",
    "time_op",
    "ToDatetimeOp",
    "ToTimestampOp",
    "StrftimeOp",
    # Numeric ops
    "abs_op",
    "add_op",
    "arccos_op",
    "arccosh_op",
    "arcsin_op",
    "arcsinh_op",
    "arctan2_op",
    "arctan_op",
    "arctanh_op",
    "ceil_op",
    "cos_op",
    "cosh_op",
    "div_op",
    "exp_op",
    "expm1_op",
    "floor_op",
    "floordiv_op",
    "ln_op",
    "log1p_op",
    "log10_op",
    "mod_op",
    "mul_op",
    "neg_op",
    "pos_op",
    "pow_op",
    "round_op",
    "sin_op",
    "sinh_op",
    "sqrt_op",
    "sub_op",
    "tan_op",
    "tanh_op",
    "unsafe_pow_op",
    # Array ops
    "ArrayIndexOp",
    "ArraySliceOp",
    "ArrayToStringOp",
    # Blob ops
    "ObjGetAccessUrl",
    "obj_make_ref_op",
    "obj_fetch_metadata_op",
    # Struct ops
    "StructFieldOp",
    "StructOp",
    # Remote Functions ops
    "BinaryRemoteFunctionOp",
    "NaryRemoteFunctionOp",
    "RemoteFunctionOp",
    # Frequency ops
    "DatetimeToIntegerLabelOp",
    "FloorDtOp",
    "IntegerLabelToDatetimeOp",
    # JSON ops
    "JSONExtract",
    "JSONExtractArray",
    "JSONExtractStringArray",
    "JSONSet",
    "ParseJSON",
    "ToJSONString",
    # Bool ops
    "and_op",
    "or_op",
    "xor_op",
    # Comparison ops
    "eq_null_match_op",
    "eq_op",
    "ge_op",
    "gt_op",
    "le_op",
    "lt_op",
    "ne_op",
    # Distance ops
    "cosine_distance_op",
    "euclidean_distance_op",
    "manhattan_distance_op",
    # Geo ops
    "geo_x_op",
    "geo_y_op",
    # Numpy ops mapping
    "NUMPY_TO_BINOP",
    "NUMPY_TO_OP",
]
