import unittest
from unittest.mock import Mock
import networkx as nx

from bigframes.functions.SchemaChangeDAG import CommandDAG

class TestCommandDAG(unittest.TestCase):
    def setUp(self):
        self.receiver = Mock()
        self.command_dCommandDAG(self.receiver)

    def test_finish_level(self):
        # Mock the DiGraph class
        mock_dag = Mock(spec=nx.DiGraph)
        self.command_dag.dag = mock_dag

        # Mock the nodes and their in/out degrees
        nodes = ['A', 'B', 'C']
        in_degrees = [1, 0, 2]
        out_degrees = [0, 0, 0]

        # Set the return values for in_degree and out_degree methods
        mock_dag.in_degree.side_effect = in_degrees
        mock_dag.out_degree.side_effect = out_degrees

        # Call the finish_level method
        self.command_dag.finish_level()

        # Assert that the cols_per_level dictionary is updated correctly
        expected_cols_per_level = {0: ['A', 'B']}
        self.assertEqual(self.command_dag.cols_per_level, expected_cols_per_level)

    def test_execute(self):
        # Mock the action and command node
        action = Mock()
        action.targets = ['target1', 'target2']
        command_node = Mock()

        # Mock the DiGraph class
        mock_dag = Mock(spec=nx.DiGraph)
        self.command_dag.dag = mock_dag

        # Add the command node to the DAG
        mock_dag.nodes.return_value = []
        mock_dag.add_node.return_value = command_node

        # Call the execute method
        self.command_dag.execute(action)

        # Assert that the command node is added to the DAG
        mock_dag.add_node.assert_called_once_with(command_node, command=action.command)

if __name__ == '__main__':
    unittest.main()