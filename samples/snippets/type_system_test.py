# Copyright 2025 Google LLC
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


def test_type_system_examples() -> None:
    # [START bigquery_dataframes_type_sytem_local_type_conversion]
    import bigframes.pandas as bpd
    import pandas as pd

    s = pd.Series([pd.Timestamp('20250101')])
    print(s.dtype)
    # datetime64[ns]
    print(bpd.read_pandas(s).dtype)
    # timestamp[us][pyarrow]
    # [END bigquery_dataframes_type_sytem_local_type_conversion]

    # [START bigquery_dataframes_type_system_pyarrow_preference]
    import bigframes.pandas as bpd
    import pandas as pd
    import datetime

    s = pd.Series([datetime.date(2025, 1, 1)])
    s + pd.Timedelta(hours=12)
    # 0	2025-01-01
    # dtype: object

    bpd.read_pandas(s) + pd.Timedelta(hours=12)
    # 0    2025-01-01 12:00:00
    # dtype: timestamp[us][pyarrow]
    # [END bigquery_dataframes_type_system_pyarrow_preference]

    # [START bigquery_dataframes_type_system_simple_json]
    import bigframes.pandas as bpd
    import db_dtypes
    import pandas as pd

    json_data = [
        "1",
        '"str"',
        "false",
        '["a", {"b": 1}, null]',
        '{"a": {"b": [1, 2, 3], "c": true}}',
        None,
    ]
    bpd.Series(json_data, dtype=pd.ArrowDtype(db_dtypes.JSONArrowType()))
    # 0                               1
    # 1                           "str"
    # 2                           false
    # 3              ["a",{"b":1},null]
    # 4    {"a":{"b":[1,2,3],"c":true}}
    # 5                            <NA>
    # dtype: extension<dbjson<JSONArrowType>>[pyarrow]
    # [END bigquery_dataframes_type_system_simple_json]

    # [START bigquery_dataframes_type_system_mixed_json]
    import bigframes.pandas as bpd
    import db_dtypes
    import pandas as pd
    import pyarrow as pa

    list_data = [
        [{"key": "1"}],
        [{"key": None}],
        [{"key": '["1","3","5"]'}],
        [{"key": '{"a":1,"b":["x","y"],"c":{"x":[],"z":false}}'}],
    ]
    pa_array = pa.array(list_data, type=pa.list_(pa.struct([("key", pa.string())])))
    bpd.Series(
        pd.arrays.ArrowExtensionArray(pa_array),
        dtype=pd.ArrowDtype(
            pa.list_(pa.struct([("key", db_dtypes.JSONArrowType())])),
        ),
    )
    # 0                                       [{'key': '1'}]
    # 1                                      [{'key': None}]
    # 2                           [{'key': '["1","3","5"]'}]
    # 3    [{'key': '{"a":1,"b":["x","y"],"c":{"x":[],"z"...
    # dtype: list<item: struct<key: extension<dbjson<JSONArrowType>>>>[pyarrow]
    # [END bigquery_dataframes_type_system_mixed_json]

    # [START bigquery_dataframes_type_system_load_timedelta]
    import bigframes.pandas as bpd
    import pandas as pd

    s = pd.Series([pd.Timedelta('1s'), pd.Timedelta('2m')])
    bpd.read_pandas(s)
    # 0    0 days 00:00:01
    # 1    0 days 00:02:00
    # dtype: duration[us][pyarrow]
    # [END bigquery_dataframes_type_system_load_timedelta]

    # [START bigquery_dataframes_type_system_timedelta_precision]
    import bigframes.pandas as bpd
    import pandas as pd

    s = pd.Series([pd.Timedelta('999ns')]).dt.round("us")
    bpd.read_pandas(s)
    # 0    0 days 00:00:00.000001
    # dtype: duration[us][pyarrow]
    # [END bigquery_dataframes_type_system_timedelta_precision]

    # [START bigquery_dataframes_type_system_cast_timedelta]
    import bigframes.pandas as bpd

    bpd.to_timedelta([1, 2, 3], unit='s')
    # 0    0 days 00:00:01
    # 1    0 days 00:00:02
    # 2    0 days 00:00:03
    # dtype: duration[us][pyarrow]
    # [END bigquery_dataframes_type_system_cast_timedelta]

    # [START bigquery_dataframes_type_system_list_accessor]
    import bigframes.pandas as bpd

    s = bpd.Series([[1, 2, 3],[4, 5],[6]])  # dtype: list<item: int64>[pyarrow]

    # Access the first elements of each list
    s.list[0]
    # 0    1
    # 1    4
    # 2    6
    # dtype: Int64

    # Get the lengths of each list
    s.list.len()
    # 0    3
    # 1    2
    # 2    1
    # dtype: Int64
    # [END bigquery_dataframes_type_system_list_accessor]

    # [START bigquery_dataframes_type_system_struct_accessor]
    import bigframes.pandas as bpd

    structs = [
        {'id': 101, 'category': 'A'},
        {'id': 102, 'category': 'B'},
        {'id': 103, 'category': 'C'},
    ]
    s = bpd.Series(structs)
    # Get the 'id' field of each struct
    s.struct.field('id')
    # 0    101
    # 1    102
    # 2    103
    # Name: id, dtype: Int64
    # [END bigquery_dataframes_type_system_struct_accessor]

    # [START bigquery_dataframes_type_system_struct_accessor_shortcut]
    import bigframes.pandas as bpd

    structs = [
        {'id': 101, 'category': 'A'},
        {'id': 102, 'category': 'B'},
        {'id': 103, 'category': 'C'},
    ]
    s = bpd.Series(structs)

    # not explicitly using the "struct" property
    s.id 
    # 0    101
    # 1    102
    # 2    103
    # Name: id, dtype: Int64
    # [END bigquery_dataframes_type_system_struct_accessor_shortcut]

    # [START bigquery_dataframes_type_system_string_accessor]
    import bigframes.pandas as bpd

    s = bpd.Series(["abc", "de", "1"]) # dtype: string[pyarrow]

    # Get the first character of each string
    s.str[0]
    # 0    a
    # 1    d
    # 2    1
    # dtype: string

    # Check whether there are only alphabetic characters in each string
    s.str.isalpha()
    # 0     True
    # 1     True
    # 2     False
    # dtype: boolean

    # Cast the alphabetic characters to their upper cases for each string
    s.str.upper()
    # 0    ABC
    # 1     DE
    # 2      1
    # dtype: string
    # [END bigquery_dataframes_type_system_string_accessor]

    # [START bigquery_dataframes_type_system_geo_accessor]
    import bigframes.pandas as bpd
    from shapely.geometry import Point

    s = bpd.Series([Point(1, 0), Point(2, 1)]) # dtype: geometry

    s.geo.y
    # 0    0.0
    # 1    1.0
    # dtype: Float64
    # [END bigquery_dataframes_type_system_geo_accessor]

    # [START bigquery_dataframes_type_system_json_query]
    import bigframes.pandas as bpd
    import bigframes.bigquery as bbq
    import db_dtypes
    import pandas as pd
    import pyarrow as pa

    fruits = [
        '{"fruits": [{"name": "apple"}, {"name": "cherry"}]}',
        '{"fruits": [{"name": "guava"}, {"name": "grapes"}]}',
    ]

    json_s = bpd.Series(fruits, dtype=pd.ArrowDtype(db_dtypes.JSONArrowType()))
    bbq.json_query(json_s, "$.fruits[0]")
    # 0    {"name":"apple"}
    # 1    {"name":"guava"}
    # dtype: extension<dbjson<JSONArrowType>>[pyarrow]
    # [END bigquery_dataframes_type_system_json_query]

    # [START bigquery_dataframes_type_system_json_query]
    import bigframes.pandas as bpd
    import bigframes.bigquery as bbq
    import db_dtypes
    import pandas as pd
    import pyarrow as pa

    fruits = [
        '{"fruits": [{"name": "apple"}, {"name": "cherry"}]}',
        '{"fruits": [{"name": "guava"}, {"name": "grapes"}]}',
    ]

    json_s = bpd.Series(fruits, dtype=pd.ArrowDtype(db_dtypes.JSONArrowType()))
    bbq.json_query(json_s, "$.fruits[0]")
    # 0    {"name":"apple"}
    # 1    {"name":"guava"}
    # dtype: extension<dbjson<JSONArrowType>>[pyarrow]
    # [END bigquery_dataframes_type_system_json_query]

    # [START bigquery_dataframes_type_system_json_extract_array]
    import bigframes.pandas as bpd
    import bigframes.bigquery as bbq
    import db_dtypes
    import pandas as pd
    import pyarrow as pa

    fruits = [
    '{"fruits": [{"name": "apple"}, {"name": "cherry"}]}',
    '{"fruits": [{"name": "guava"}, {"name": "grapes"}]}',
    ]

    json_s = bpd.Series(fruits, dtype=pd.ArrowDtype(db_dtypes.JSONArrowType()))

    bbq.json_extract_array(json_s, "$.fruits")
    # 0    ['{"name":"apple"}' '{"name":"cherry"}']
    # 1    ['{"name":"guava"}' '{"name":"grapes"}']
    # dtype: list<item: extension<dbjson<JSONArrowType>>>[pyarrow]
    # [END bigquery_dataframes_type_system_json_extract_array]
