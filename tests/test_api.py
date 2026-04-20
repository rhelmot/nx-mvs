from __future__ import annotations

from pathlib import Path
import unittest

import networkx as nx

from mvs import enumerate_convex_subgraphs, enumerate_maximum_convex_subgraphs


REPO_ROOT = Path(__file__).resolve().parents[1]


def read_dimacs_graph(path: Path, *, weighted: bool) -> nx.DiGraph[int]:
    graph: nx.DiGraph[int] = nx.DiGraph()
    with path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            fields = line.split(" ")
            if fields[0] == "p":
                node_count = int(fields[2])
                graph.graph["name"] = fields[4]
                graph.graph["frequency"] = int(fields[5])
                for node in range(node_count):
                    graph.add_node(node, weight=1.0, forbidden=False)
            elif fields[0] == "e":
                graph.add_edge(int(fields[1]) - 1, int(fields[2]) - 1)
            elif fields[0] == "n":
                node = int(fields[1]) - 1
                if weighted:
                    graph.nodes[node]["weight"] = float(fields[2])
                graph.nodes[node]["forbidden"] = fields[3] == "1"
    return graph


class TestMVS(unittest.TestCase):
    def test_maximum_enumeration_with_opted_in_sources_and_sinks(self) -> None:
        graph = nx.DiGraph()
        graph.add_node("sink", forbidden=True)
        graph.add_edges_from(
            [
                ("src", "mid"),
                ("mid", "sink"),
            ]
        )

        result = {
            frozenset(nodes)
            for nodes in enumerate_maximum_convex_subgraphs(
                graph,
                1,
                1,
                forbid_sources_and_sinks=False,
            )
        }
        self.assertSetEqual({frozenset({"src", "mid"})}, result)

    def test_maximum_enumeration_with_opted_in_sink(self) -> None:
        graph = nx.DiGraph()
        graph.add_node("src", forbidden=True)
        graph.add_edges_from(
            [
                ("src", "mid"),
                ("mid", "sink"),
            ]
        )

        result = {
            frozenset(nodes)
            for nodes in enumerate_maximum_convex_subgraphs(
                graph,
                1,
                1,
                forbid_sources_and_sinks=False,
                allow_zero_outputs=True,
            )
        }
        self.assertSetEqual({frozenset({"mid", "sink"})}, result)

    def test_maximum_enumeration_can_include_zero_outputs_when_enabled(self) -> None:
        graph = nx.DiGraph()
        graph.add_node("src", forbidden=True)
        graph.add_node("mid", weight=1.0)
        graph.add_node("sink", weight=5.0)
        graph.add_edges_from(
            [
                ("src", "mid"),
                ("mid", "sink"),
            ]
        )

        result = {
            frozenset(nodes)
            for nodes in enumerate_maximum_convex_subgraphs(
                graph,
                1,
                1,
                weighted=True,
                forbid_sources_and_sinks=False,
                allow_zero_outputs=True,
            )
        }
        self.assertSetEqual({frozenset({"mid", "sink"})}, result)

    def test_sources_and_sinks_are_forbidden_by_default(self) -> None:
        graph = nx.DiGraph()
        graph.add_node("sink", forbidden=True)
        graph.add_edges_from(
            [
                ("src", "mid"),
                ("mid", "sink"),
            ]
        )

        result = {frozenset(nodes) for nodes in enumerate_convex_subgraphs(graph, 1, 1)}
        self.assertSetEqual(
            {
                frozenset({"src"}),
                frozenset({"mid"}),
            },
            result,
        )

    def test_sources_and_sinks_can_be_opted_in(self) -> None:
        graph = nx.DiGraph()
        graph.add_node("sink", forbidden=True)
        graph.add_edges_from(
            [
                ("src", "mid"),
                ("mid", "sink"),
            ]
        )

        result = {
            frozenset(nodes)
            for nodes in enumerate_convex_subgraphs(
                graph,
                1,
                1,
                forbid_sources_and_sinks=False,
            )
        }
        self.assertSetEqual(
            {
                frozenset({"src"}),
                frozenset({"mid"}),
                frozenset({"src", "mid"}),
            },
            result,
        )

    def test_exhaustive_enumeration_works_without_explicit_forbidden_nodes(self) -> None:
        graph = nx.DiGraph()
        graph.add_edges_from(
            [
                ("a", "b"),
            ]
        )

        result = {
            frozenset(nodes)
            for nodes in enumerate_convex_subgraphs(
                graph,
                1,
                1,
                forbid_sources_and_sinks=False,
            )
        }
        self.assertIn(frozenset({"a"}), result)

    def test_exhaustive_enumeration_can_include_zero_outputs_when_enabled(self) -> None:
        graph = nx.DiGraph()
        graph.add_node("src", forbidden=True)
        graph.add_edges_from(
            [
                ("src", "mid"),
                ("mid", "sink"),
            ]
        )

        without_zero_outputs = {
            frozenset(nodes)
            for nodes in enumerate_convex_subgraphs(
                graph,
                1,
                1,
                forbid_sources_and_sinks=False,
            )
        }
        with_zero_outputs = {
            frozenset(nodes)
            for nodes in enumerate_convex_subgraphs(
                graph,
                1,
                1,
                forbid_sources_and_sinks=False,
                allow_zero_outputs=True,
            )
        }
        self.assertNotIn(frozenset({"mid", "sink"}), without_zero_outputs)
        self.assertIn(frozenset({"mid", "sink"}), with_zero_outputs)

    def test_exhaustive_enumeration_returns_non_maximum_subgraphs(self) -> None:
        graph = nx.DiGraph()
        graph.add_node("src", forbidden=True)
        graph.add_node("a")
        graph.add_node("b")
        graph.add_node("sink", forbidden=True)
        graph.add_edges_from(
            [
                ("src", "a"),
                ("a", "b"),
                ("b", "sink"),
            ]
        )

        result = {frozenset(nodes) for nodes in enumerate_convex_subgraphs(graph, 1, 1)}
        self.assertSetEqual(
            {
                frozenset({"a"}),
                frozenset({"b"}),
                frozenset({"a", "b"}),
            },
            result,
        )

    def test_repository_benchmarks(self) -> None:
        cases = [
            ("mvs/data/DFG_crypt_Transform_entry.45.txt", 1, 1, 64),
            ("mvs/data/DFG_crypt_Transform_entry.45.txt", 2, 2, 14),
            ("mvs/data/DFG_hadamard_HadamardSAD8x8_for.body.1.txt", 18, 18, 1),
            ("mvs/data/DFG_hadamard_HadamardSAD8x8_for.body.1.txt", 17, 17, 1),
            ("mvs/data/DFG_hadamard_HadamardSAD8x8_for.body.1.txt", 16, 16, 1),
            ("mvs/data/DFG_hadamard_HadamardSAD8x8_for.body.1.txt", 15, 15, 16),
            ("mvs/data/DFG_hadamard_HadamardSAD8x8_for.body.1.txt", 14, 14, 8),
        ]
        for relative_path, max_inputs, max_outputs, expected_count in cases:
            with self.subTest(path=relative_path, max_inputs=max_inputs, max_outputs=max_outputs):
                graph = read_dimacs_graph(REPO_ROOT / relative_path, weighted=False)
                result = list(
                    enumerate_maximum_convex_subgraphs(
                        graph,
                        max_inputs,
                        max_outputs,
                    )
                )
                self.assertEqual(expected_count, len(result))


if __name__ == "__main__":
    unittest.main()
