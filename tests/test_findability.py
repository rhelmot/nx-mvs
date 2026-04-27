from __future__ import annotations

import unittest

import networkx as nx

from findability import measure_findability


class TestFindability(unittest.TestCase):
    def test_measure_findability_reports_exact_and_random_growth_steps(self) -> None:
        graph = nx.DiGraph()
        graph.add_node("p", forbidden=True)
        graph.add_edges_from(
            [
                ("p", "a"),
                ("p", "b"),
                ("p", "c"),
                ("c", "b"),
            ]
        )

        result = measure_findability(
            graph,
            target_nodes={"a", "b", "c"},
            max_num_inputs=2,
            max_subgraph_size=10,
            forbid_sources_and_sinks=False,
            max_states_expanded=0,
            max_samples=0,
            max_children_per_state=1,
            size_bin_width=1,
            exact_kernel_size=1,
            forbidden_attr=None,
            body_forbidden_attr="forbidden",
        )

        self.assertTrue(result.found)
        self.assertEqual(1, result.minimum_growth_steps)
        self.assertEqual(frozenset({"a"}), result.minimum_growth_seed)
        expected_random_growth_steps = result.expected_random_growth_steps
        self.assertIsNotNone(expected_random_growth_steps)
        assert expected_random_growth_steps is not None
        self.assertAlmostEqual(1.5, expected_random_growth_steps)
        self.assertEqual(frozenset({"a"}), result.expected_random_seed)
        self.assertAlmostEqual(1.0, result.random_search_success_probability)
        self.assertEqual(2, result.sample_count)
        self.assertEqual(2, result.launch_point_count)

    def test_measure_findability_rejects_invalid_target(self) -> None:
        graph = nx.DiGraph()
        graph.add_node("p", forbidden=True)
        graph.add_edges_from(
            [
                ("p", "a"),
                ("p", "b"),
                ("p", "c"),
                ("c", "b"),
            ]
        )

        with self.assertRaises(ValueError):
            measure_findability(
                graph,
                target_nodes={"a", "c"},
                max_num_inputs=2,
                max_subgraph_size=10,
                forbid_sources_and_sinks=False,
                max_states_expanded=0,
                max_samples=0,
                max_children_per_state=1,
                size_bin_width=1,
                exact_kernel_size=1,
                forbidden_attr=None,
                body_forbidden_attr="forbidden",
            )

    def test_measure_findability_respects_alternate_graph(self) -> None:
        graph = nx.DiGraph()
        graph.add_node("p", forbidden=True)
        graph.add_edges_from(
            [
                ("p", "a"),
                ("p", "b"),
                ("p", "c"),
            ]
        )
        alternate_graph = nx.DiGraph()
        alternate_graph.add_node("p", forbidden=True)
        alternate_graph.add_edges_from(
            [
                ("p", "a"),
                ("p", "b"),
                ("p", "c"),
                ("a", "b"),
                ("b", "c"),
            ]
        )

        with self.assertRaises(ValueError):
            measure_findability(
                graph,
                alternate_graph,
                target_nodes={"a", "c"},
                max_num_inputs=1,
                max_subgraph_size=10,
                forbid_sources_and_sinks=False,
                max_states_expanded=0,
                max_samples=0,
                max_children_per_state=1,
                size_bin_width=1,
                exact_kernel_size=1,
                forbidden_attr=None,
                body_forbidden_attr="forbidden",
            )
