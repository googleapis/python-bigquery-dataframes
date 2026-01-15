# Copyright 2026 Google LLC
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


from bigframes import dtypes


def _add_data_type(existing_types: int, curr_type: dtypes.Dtype) -> int:
    return existing_types | _get_dtype_mask(curr_type)


def _get_dtype_mask(dtype: dtypes.Dtype) -> int:
    match dtype:
        case dtypes.INT_DTYPE:
            return 1 << 1
        case dtypes.FLOAT_DTYPE:
            return 1 << 2
        case dtypes.BOOL_DTYPE:
            return 1 << 3
        case dtypes.STRING_DTYPE:
            return 1 << 4
        case dtypes.BYTES_DTYPE:
            return 1 << 5
        case dtypes.DATE_DTYPE:
            return 1 << 6
        case dtypes.TIME_DTYPE:
            return 1 << 7
        case dtypes.DATETIME_DTYPE:
            return 1 << 8
        case dtypes.TIMESTAMP_DTYPE:
            return 1 << 9
        case dtypes.TIMEDELTA_DTYPE:
            return 1 << 10
        case dtypes.NUMERIC_DTYPE:
            return 1 << 11
        case dtypes.BIGNUMERIC_DTYPE:
            return 1 << 12
        case dtypes.GEO_DTYPE:
            return 1 << 13
        case dtypes.JSON_DTYPE:
            return 1 << 14

    if dtypes.is_struct_like(dtype):
        mask = 1 << 15
        if dtype == dtypes.OBJ_REF_DTYPE:
            # obj_ref is a special struct type for multi-modal data.
            # It should be double counted as both "struct" and its own type.
            mask = mask | (1 << 17)
        return mask

    if dtypes.is_array_like(dtype):
        return 1 << 16

    # If an unknown datat type is present, mark it with the least significant bit.
    return 1 << 0
