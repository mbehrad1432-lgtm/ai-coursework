"""Regression checks against notebook definitions, without running its demos."""
import ast
import json
from pathlib import Path
import random
import unittest

import matplotlib
matplotlib.use('Agg')

ROOT = Path(__file__).resolve().parents[1]
namespace = {}
notebook = json.loads((ROOT / 'search_and_optimization.ipynb').read_text())
for cell in notebook['cells']:
    if cell['cell_type'] != 'code':
        continue
    tree = ast.parse(''.join(cell['source']))
    definitions = [node for node in tree.body
                   if isinstance(node, (ast.Import, ast.ImportFrom, ast.FunctionDef, ast.ClassDef))]
    exec(compile(ast.Module(body=definitions, type_ignores=[]), '<notebook definitions>', 'exec'), namespace)

class SearchChecks(unittest.TestCase):
    def test_shortest_paths_and_route_validity(self):
        maze = namespace['Maze']([[0, 1, 0], [0, 0, 0], [0, 1, 0]])
        for algorithm in ['dfs', 'bfs', 'a_star']:
            path, expanded = namespace[algorithm](maze, (0, 0), (2, 2))
            self.assertEqual((path[0], path[-1]), ((0, 0), (2, 2)))
            self.assertTrue(all(b in maze.get_neighbors(a) for a, b in zip(path, path[1:])))
            self.assertGreaterEqual(expanded, len(path))
            if algorithm != 'dfs':
                self.assertEqual(len(path) - 1, 4)

    def test_unreachable_goal_counts_all_reachable_states(self):
        # A four-state cyclic component and an isolated goal expose DFS's
        # previous branch-local visited sets and incorrect expansion counter.
        maze = namespace['Maze']([[0, 0, 1], [0, 0, 1], [1, 1, 0]])
        for algorithm in ['dfs', 'bfs', 'a_star']:
            path, expanded = namespace[algorithm](maze, (0, 0), (2, 2))
            self.assertEqual(path, [])
            self.assertEqual(expanded, 4)

    def test_start_equals_goal(self):
        maze = namespace['Maze']([[0]])
        for algorithm in ['dfs', 'bfs', 'a_star']:
            self.assertEqual(namespace[algorithm](maze, (0, 0), (0, 0)), ([(0, 0)], 1))

    def test_bounded_search_and_best_point_retention(self):
        # A large step and exclusively random acceptance force the old
        # unbounded implementation beyond the permitted interval.
        for seed in range(20):
            random.seed(seed)
            x, value, path = namespace['hill_climbing_1D_random'](
                namespace['f'], 0.4, step_size=3, p=1, max_iters=20, max_restarts=3)
            self.assertTrue(all(-2 <= point <= 2 for point in path))
            self.assertEqual(x, path[-1])
            self.assertAlmostEqual(value, namespace['f'](x))
            self.assertLessEqual(value, namespace['f'](0.4))

    def test_genetic_algorithm_permutation_and_elitism(self):
        cities = namespace['getCity'](ROOT / 'TSP51.txt')
        self.assertEqual(len(cities), 51)
        for seed in range(3):
            random.seed(seed)
            population, initial = namespace['selectPopulation'](cities, 40)
            answer, _ = namespace['geneticAlgorithm'](population, len(cities), 4, 1, 1, 0, ITERATION=5)
            self.assertEqual(sorted(answer[1]), sorted(cities))
            self.assertAlmostEqual(answer[0], namespace['calcDistance'](answer[1]))
            self.assertLessEqual(answer[0], initial[0])

    def test_seeded_repeatability(self):
        def run():
            random.seed(42)
            return namespace['hill_climbing_1D_random'](namespace['f'], 0.4)
        self.assertEqual(run(), run())

if __name__ == '__main__':
    unittest.main(verbosity=2)
