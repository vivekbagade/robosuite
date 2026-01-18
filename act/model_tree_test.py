import unittest
from model_tree import ModelTree

class TestModelTree(unittest.TestCase):
    def setUp(self):
        self.tree = ModelTree()
        # Create a larger graph with multiple branches and edge types
        versions = [
            '1.0.0', '1.1.0', '1.2.0', '2.0.0', '2.1.0', '2.2.0', '3.0.0', '3.1.0'
        ]
        for v in versions:
            self.tree.add_new_version(v)

        # Episode edges
        self.tree.add_episode_edge('1.0.0', '1.1.0')
        self.tree.add_episode_edge('1.1.0', '1.2.0')
        self.tree.add_episode_edge('1.2.0', '2.0.0')
        self.tree.add_episode_edge('2.0.0', '2.1.0')
        self.tree.add_episode_edge('2.1.0', '2.2.0')
        self.tree.add_episode_edge('2.2.0', '3.0.0')
        self.tree.add_episode_edge('3.0.0', '3.1.0')

        # Weight edges (cross-links)
        self.tree.add_weight_edge('1.0.0', '2.0.0')
        self.tree.add_weight_edge('2.0.0', '3.0.0')
        self.tree.add_weight_edge('1.1.0', '2.1.0')
        self.tree.add_weight_edge('2.1.0', '3.1.0')

        # Add some metadata for completeness
        for v in versions:
            self.tree.set_node_metadata(v, {'desc': f'Version {v}'})

    def test_walk_back_single_step(self):
        result = self.tree._walk_back('3.1.0', edge_type='episode', lookback=1)
        self.assertIn('3.0.0', result)
        self.assertEqual(len(result), 1)

    def test_walk_back_two_steps(self):
        result = self.tree._walk_back('3.1.0', edge_type='episode', lookback=2)
        self.assertIn('3.0.0', result)
        self.assertIn('2.2.0', result)
        self.assertEqual(len(result), 2)

    def test_walk_back_weight_edges(self):
        result = self.tree._walk_back('3.1.0', edge_type='weight', lookback=2)
        self.assertIn('2.1.0', result)
        self.assertIn('1.1.0', result)
        self.assertEqual(len(result), 2)

    def test_walk_back_no_node(self):
        result = self.tree._walk_back('nonexistent', edge_type='episode', lookback=1)
        self.assertEqual(result, [])

if __name__ == '__main__':
    unittest.main()