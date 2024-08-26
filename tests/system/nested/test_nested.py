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

#from google.cloud import bigquery
#from bigframes.functions.nested import BQSchemaLayout, SchemaField

#from google.cloud.bigquery_storage_v1 import types as gtypes
#import pytest
#from typing import List

import bigframes.pandas as bfpd
import bigframes.core as core
from bigframes.core.schema_tracking import set_project
from bigframes.dataframe import DataFrame
#from bigframes.core.nodes import NestedDataContextManager
#from bigframes.core import Session


# start context manager (cm) in pandas/__init__.py
# use dataframe object, there is dtypes info on it.
# cm constructur get schema by: dataframe._cashed [replaces block by cached version, one to one bq table to dataframe]
#   and get block with _block.expr

# def table_schema(table_name_full: str) -> List[SchemaField]:
#     project = table_name_full.split(".")[0]
#     client = bigquery.Client(project=project, location="EU")
#     query_job = client.get_table(table_name_full)
#     return query_job.schema


# def test_unroll_schema():  #table_name_full: pytest.CaptureFixture[str]
#     schema = table_schema("gmbigframes.nested.tiny") # "vf-de-aib-prd-cmr-chn-lab.staging.scs_mini")
#     bqs = BQSchemaLayout(schema)
#     bqs.determine_layout() # TODO: add prefix get_ or determine_
#     return bqs
#     #assert isinstance(schema, List[SchemaField])

# def test_nested_cm():
#     bfpd.options.bigquery.project = "gmbigframes"
#     bfpd.options.bigquery.location = "EU"


# def fct_cm(cm: NestedDataFrame):
#     cm._current_data = bfpd.read_gbq(f"SELECT * FROM {table}"),
#     testdf.apply(cm._current_data),
#     bfpd.get_dummies(cm._current_data)   

def create_simple_nested():
    """
            import bigframes.pandas as bpd
          import pyarrow as pa
           bpd.options.display.progress_bar = None
          s = bpd.Series(
            ...     [
            ...         {"version": 1, "project": "pandas"},
            ...         {"version": 2, "project": "pandas"},
            ...         {"version": 1, "project": "numpy"},
            ...     ],
            ...     dtype=bpd.ArrowDtype(pa.struct(
            ...         [("version", pa.int64()), ("project", pa.string())]
            ...     ))
            ... ) """
    import pyarrow as pa
    import pandas as pd
    s = pd.Series([
            {"version": 1, "project": "pandas"},
            {"version": 2, "project": "pandas"},
            {"version": 1, "project": "numpy"},
        ], dtype=pd.ArrowDtype( #pa.struct([("sel", 
                                           pa.struct([("version", pa.int64()), ("project", pa.string())]
                                ))
        #])
        )
    #)
    dfp = pd.DataFrame(s.to_frame())
    
    d = DataFrame(dfp, index=[])
    d.to_gbq("andreas_beschorner.nested_dbg")
   
    # import pandas as pd
    # df1 = pd.DataFrame({
    #     "aa": [2, 3, 4],
    #     "bb": ["I", "you", "we"],
    #     "cc": {
    #         "k1": [[22], [44], [33]],
    #         "k2": [["one"], ["2"], [None]]
    #     } 
    # }, index=None)
    # df1.to_gbq("andreas_beschorner.nested_dbg")

if __name__ == "__main__":
    #TODO: autodetect if bfpd is already setup and copy proj/loc if availabe
    #set_project(project="gmbigframes", location="europe-west3")
    #table = "gmbigframes.nested.tiny"  #"vf-de-aib-prd-cmr-chn-lab.staging.scs_mini"
    set_project(project="vf-de-ca-lab", location="europe-west3")
    table="andreas_beschorner.nested_tiny"
    create_simple_nested()
    exit(0)
    #testdf = DataFrame({"a": [1]}, index=None)

    with core.nested_data_context_manager:
        df = bfpd.read_gbq(f"SELECT * FROM {table} limit 10")
        df_n = df.explode_nested(sep_explode=core.nested_data_context_manager.sep_layers)
        df = df.rename(columns={"event_sequence.POSO": "event_sequence.pso"})
        pass
        #testdf = DataFrame({"ooc_flag": [1], "test_value": ["Grmph"]}, index=None)
        #TODO: How create 

        #ncm |=  ncm.data, {"columns": []} | n_get_dummies
    pass


    # bqs = test_unroll_schema()
    # shdl = SchemaHandler(bqs, layer_separator=bsq.layer_separator)
    # cmd = CommandDAG(shdl)
    # cmd.dag_from_schema()
    