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




# schema handler
from google.cloud.bigquery.schema import SchemaField
from google.cloud.bigquery_storage_v1 import types

from bigframes.functions.nested import BQSchemaLayout

from typing import List, Dict
from dataclasses import dataclass
from collections import defaultdict, deque

"""
    We want to facilitate a DAG for handling and executing schema changes/transitions, applying the
    "Command" pattern. The DAG will be used to handle the changing names, always relating column names to
    their complete history.

    The command bridge itself needs to communicate between schema and DAG using respective commands.

    IMPORTANT: Within a block of actions for a nested table, one should avoid parallel commands on tables.
    We do not want simultaneous schema changes of whatever column(s)
    What we need to do: Know which children (and subchildren) belong to which parents, grandparents etc.,
    so can at the end merge can back into the latter and retain the correct structure.
"""






# SchemaHandler: Receiver of commands. No (abstract) base class necessary as we have to furher receivers.





class CommandDAG(CommandBase):
    
    
    def __init__(self, receiver: SchemaHandler):
        super().__init__(receiver)
        self.dag = DiGraph()
        self.dag.add_node(self._base_root_name, node_type=NodeInfo.T_root)
        self.dag_from_schema()
        self.cols_per_level = {}  # store all cols here, level as key
        self._level = 0  # depth of DAG

    def _leaves(self):
        return [node for node in self.dag.nodes if self.dag.out_degree(node) == 0]

    # two identical properties, depending on what meaning you prefer
    @property
    def leaves(self):
        return self._leaves()

    @property
    def final_columns(self):
        return self._leaves()

    def finish_level(self):
        # save leaves
        self.cols_per_level.update({
            self._level: self._leaves()
        })
        super().finish_level()

    def dag_from_schema(self):
        schema = self.receiver.orig_schema()
        
        root_layer = True
        for layer in self.receiver.bfs(schema):
            if root_layer:
                col_type = NodeInfo.T_root
            for col_name in layer:
                assert(self.receiver.sep_layers not in col_name)
                if not root_layer:
                    col_type = self.receiver.schema_orig.map_to_type[col_name]
                    parent = col_name.rsplit(self.receiver.sep_structs, 1)[0]
                else:
                    parent = self._base_root_name
                # replace struct separator with layer separator, as struct separator must not be used in exploded column names
                col_name = col_name.replace(self.receiver.sep_structs, self.receiver.sep_layers)
                self.dag.add_node(col_name, node_type=col_type)
                self.dag.add_edge(parent, col_name)
            root_layer = False
        return

    def execute(self, action: ActionBase):
        # execute receiver command, which must return a string to list dict with schema changes
        # expand dag by changes. this results in a final parent-child-... relationship. keeping
        #  track of which nodes belong to which nests and subnests
        assert(len(set(action.targets)) == len(action.targets))  # no repeated targets
        successors = self.receiver.perform_action()
        
        # add finals schema to dag. If multiple edges 
        for cols_target, cols_src in successors.items():
            for _target in cols_target:
                # TODO: node type needs to be set to record or feature,
                # as we need to now whether it is a record parent node or whether schema changes happened
                self.dag.add_node(_target, node_type="")
                for _src in colc_src:
                    self.dag.add_edge(_src, _target)
        
    