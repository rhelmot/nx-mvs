from __future__ import annotations

from collections.abc import Callable, Hashable, Iterable, Iterator
from os import PathLike
from pathlib import Path
import unittest
from unittest.mock import patch

import networkx as nx

import mvs._api as mvs_api
from mvs import (
    ConvexSubgraphQuery,
    graph_to_input,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
BODY_ONLY_FORBIDDEN = {"forbidden_attr": None, "body_forbidden_attr": "forbidden"}
GraphSource = nx.DiGraph | str | PathLike[str]
QUERY_CONFIG_KEYS = {
    "max_subgraph_size",
    "weighted",
    "weight_attr",
    "forbidden_attr",
    "body_forbidden_attr",
    "input_forbidden_attr",
    "forbid_sources_and_sinks",
    "connected_only",
    "alternate_connected_only",
    "max_queue_size",
    "iteration_type",
    "flags",
    "ordering",
}
SAMPLING_NAMES = {
    "max_states_expanded": "sampling_max_states_expanded",
    "max_samples": "sampling_max_samples",
    "max_children_per_state": "sampling_max_children_per_state",
    "size_bin_width": "sampling_size_bin_width",
    "thicken_radius": "sampling_thicken_radius",
    "bucket_by_num_inputs": "sampling_bucket_by_num_inputs",
    "bucket_by_num_outputs": "sampling_bucket_by_num_outputs",
    "minimal_node_bin_width": "sampling_minimal_node_bin_width",
    "boundary_pair_samples": "sampling_boundary_pair_samples",
    "sampling_passes": "sampling_passes",
    "exact_kernel_size": "sampling_exact_kernel_size",
    "max_work": "sampling_max_work",
}


def _split_query_kwargs(kwargs: dict[str, object]) -> tuple[dict[str, object], dict[str, object]]:
    query_kwargs = {
        SAMPLING_NAMES.get(name, name): value
        for name, value in kwargs.items()
    }
    constructor_kwargs = {
        name: value
        for name, value in query_kwargs.items()
        if name in QUERY_CONFIG_KEYS or name.startswith("sampling_")
    }
    method_kwargs = {
        name: value
        for name, value in query_kwargs.items()
        if name not in constructor_kwargs
    }
    return constructor_kwargs, method_kwargs


def enumerate_convex_subgraphs(
    graph: nx.DiGraph,
    max_num_inputs: int | None = None,
    max_num_outputs: int | None = None,
    **kwargs: object,
) -> Iterator[set[Hashable]]:
    if max_num_inputs is None:
        max_num_inputs = kwargs.pop("max_num_inputs")  # type: ignore[assignment]
    if max_num_outputs is None:
        max_num_outputs = kwargs.pop("max_num_outputs")  # type: ignore[assignment]
    sampling = kwargs.pop("sampling", None)
    constructor_kwargs, method_kwargs = _split_query_kwargs(kwargs)
    return ConvexSubgraphQuery(**constructor_kwargs).enumerate(
        graph,
        max_num_inputs=max_num_inputs,
        max_num_outputs=max_num_outputs,
        **method_kwargs,
        sampling=sampling,  # type: ignore[arg-type]
    )


def enumerate_maximum_convex_subgraphs(
    graph: nx.DiGraph,
    max_num_inputs: int | None = None,
    max_num_outputs: int | None = None,
    **kwargs: object,
) -> Iterator[set[Hashable]]:
    if max_num_inputs is None:
        max_num_inputs = kwargs.pop("max_num_inputs")  # type: ignore[assignment]
    if max_num_outputs is None:
        max_num_outputs = kwargs.pop("max_num_outputs")  # type: ignore[assignment]
    sampling = kwargs.pop("sampling", None)
    constructor_kwargs, method_kwargs = _split_query_kwargs(kwargs)
    return ConvexSubgraphQuery(**constructor_kwargs).maximum(
        graph,
        max_num_inputs=max_num_inputs,
        max_num_outputs=max_num_outputs,
        **method_kwargs,
        sampling=sampling,  # type: ignore[arg-type]
    )


def sample_zero_output_convex_subgraphs(
    graph: nx.DiGraph,
    max_num_inputs: int,
    **kwargs: object,
) -> Iterator[set[Hashable]]:
    constructor_kwargs, method_kwargs = _split_query_kwargs(kwargs)
    return ConvexSubgraphQuery(**constructor_kwargs).sample(
        graph,
        max_num_inputs=max_num_inputs,
        max_num_outputs=0,
        **method_kwargs,
    )


def _load_graph(source: GraphSource | None) -> nx.DiGraph | None:
    if source is None:
        return None
    if isinstance(source, nx.DiGraph):
        return source
    return nx.DiGraph(nx.nx_pydot.read_dot(Path(source)))


def grow_zero_output_convex_subgraphs(
    graph: GraphSource = "graph.dot",
    alternate_graph: GraphSource | None = "graph-alt.dot",
    *,
    seed_nodes: Iterable[Hashable],
    oracle: Callable[..., object | None] | None = None,
    initial_oracle_state: object | None = None,
    max_num_inputs: int = 4,
    max_subgraph_size: int = 50,
    forbid_sources_and_sinks: bool = False,
    forbidden_attr: str | None = "forbidden",
    body_forbidden_attr: str | None = None,
    input_forbidden_attr: str | None = None,
) -> Iterator[set[Hashable]]:
    loaded_graph = _load_graph(graph)
    assert loaded_graph is not None
    yield from ConvexSubgraphQuery(
        max_subgraph_size=max_subgraph_size,
        forbidden_attr=forbidden_attr,
        body_forbidden_attr=body_forbidden_attr,
        input_forbidden_attr=input_forbidden_attr,
        forbid_sources_and_sinks=forbid_sources_and_sinks,
    ).grow(
        loaded_graph,
        set(seed_nodes),
        max_num_inputs=max_num_inputs,
        max_num_outputs=0,
        alternate_graph=_load_graph(alternate_graph),
        oracle=oracle,
        initial_oracle_state=initial_oracle_state,
    )


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
    def test_query_object_matches_function_api(self) -> None:
        graph = nx.DiGraph()
        graph.add_node("src", forbidden=True)
        graph.add_node("sink", forbidden=True)
        graph.add_edges_from(
            [
                ("src", "a"),
                ("a", "b"),
                ("b", "sink"),
            ]
        )

        query = ConvexSubgraphQuery(
            **BODY_ONLY_FORBIDDEN,
            sampling_max_states_expanded=0,
            sampling_max_samples=2,
            sampling_boundary_pair_samples=2,
        )

        self.assertSetEqual(
            {
                frozenset(nodes)
                for nodes in enumerate_convex_subgraphs(
                    graph,
                    1,
                    1,
                    sampling=False,
                    **BODY_ONLY_FORBIDDEN,
                )
            },
            {
                frozenset(nodes)
                for nodes in query.enumerate(
                    graph,
                    1,
                    1,
                    sampling=False,
                )
            },
        )
        self.assertSetEqual(
            {
                frozenset(nodes)
                for nodes in enumerate_maximum_convex_subgraphs(
                    graph,
                    1,
                    1,
                    sampling=False,
                    **BODY_ONLY_FORBIDDEN,
                )
            },
            {
                frozenset(nodes)
                for nodes in query.maximum(
                    graph,
                    1,
                    1,
                    sampling=False,
                )
            },
        )
        self.assertSetEqual(
            {frozenset({"a"}), frozenset({"b"})},
            {
                frozenset(nodes)
                for nodes in query.sample(
                    graph,
                    1,
                    1,
                )
            },
        )

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
                **BODY_ONLY_FORBIDDEN,
            )
        }
        self.assertSetEqual({frozenset({"mid", "sink"})}, result)

    def test_exhaustive_enumeration_with_opted_in_sink_successor_of_output(self) -> None:
        graph = nx.DiGraph()
        graph.add_edges_from(
            [
                ("output", "internal"),
                ("output", "external"),
                ("input", "internal"),
                ("internal", "sink"),
            ]
        )

        result = {
            frozenset(nodes)
            for nodes in enumerate_convex_subgraphs(
                graph,
                1,
                1,
                forbid_sources_and_sinks=False,
                connected_only=True,
                sampling=False,
            )
        }
        self.assertIn(frozenset({"output", "internal", "sink"}), result)

    def test_exhaustive_enumeration_with_output_reaching_forbidden_external_node(self) -> None:
        graph = nx.DiGraph()
        graph.add_node("forbidden", forbidden=True)
        graph.add_edges_from(
            [
                ("input", "output"),
                ("output", "internal"),
                ("other", "internal"),
                ("internal", "sink"),
                ("output", "forbidden"),
            ]
        )

        result = {
            frozenset(nodes)
            for nodes in enumerate_convex_subgraphs(
                graph,
                1,
                1,
                forbid_sources_and_sinks=False,
                connected_only=True,
                sampling=False,
                **BODY_ONLY_FORBIDDEN,
            )
        }
        self.assertIn(frozenset({"output", "other", "internal", "sink"}), result)

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
                **BODY_ONLY_FORBIDDEN,
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
                allow_zero_outputs=True,
                **BODY_ONLY_FORBIDDEN,
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

    def test_body_forbidden_nodes_may_still_appear_as_inputs(self) -> None:
        graph = nx.DiGraph()
        graph.add_node("i", body_forbidden=True)
        graph.add_node("o", body_forbidden=True)
        graph.add_edges_from(
            [
                ("i", "a"),
                ("a", "o"),
            ]
        )

        result = {
            frozenset(nodes)
            for nodes in enumerate_convex_subgraphs(
                graph,
                1,
                1,
                body_forbidden_attr="body_forbidden",
                forbid_sources_and_sinks=False,
            )
        }

        self.assertIn(frozenset({"a"}), result)
        self.assertNotIn(frozenset({"i"}), result)

    def test_input_forbidden_nodes_may_still_appear_as_body(self) -> None:
        graph = nx.DiGraph()
        graph.add_node("i", input_forbidden=True)
        graph.add_edges_from(
            [
                ("i", "a"),
            ]
        )

        result = {
            frozenset(nodes)
            for nodes in enumerate_convex_subgraphs(
                graph,
                1,
                1,
                input_forbidden_attr="input_forbidden",
                forbid_sources_and_sinks=False,
                allow_zero_outputs=True,
            )
        }

        self.assertNotIn(frozenset({"a"}), result)
        self.assertIn(frozenset({"i", "a"}), result)

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

    def test_exhaustive_enumeration_excludes_zero_output_results_by_default(self) -> None:
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

        self.assertSetEqual({frozenset({"a"})}, result)

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
                **BODY_ONLY_FORBIDDEN,
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
                **BODY_ONLY_FORBIDDEN,
            )
        }
        self.assertNotIn(frozenset({"mid", "sink"}), without_zero_outputs)
        self.assertIn(frozenset({"mid", "sink"}), with_zero_outputs)

    def test_direct_zero_output_enumeration(self) -> None:
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
            for nodes in enumerate_convex_subgraphs(
                graph,
                1,
                0,
                forbid_sources_and_sinks=False,
                **BODY_ONLY_FORBIDDEN,
            )
        }
        self.assertSetEqual({frozenset({"mid", "sink"})}, result)

    def test_sample_zero_output_convex_subgraphs_is_deterministic(self) -> None:
        graph = nx.DiGraph()
        graph.add_node("p", forbidden=True)
        graph.add_edges_from(
            [
                ("p", "a"),
                ("p", "b"),
                ("p", "c"),
                ("p", "d"),
            ]
        )

        sample = [
            frozenset(nodes)
            for nodes in sample_zero_output_convex_subgraphs(
                graph,
                1,
                forbid_sources_and_sinks=False,
                max_states_expanded=32,
                max_samples=4,
                max_children_per_state=1,
                size_bin_width=1,
                **BODY_ONLY_FORBIDDEN,
            )
        ]

        self.assertEqual(
            [
                frozenset({"a"}),
                frozenset({"a", "b"}),
                frozenset({"a", "b", "c"}),
                frozenset({"a", "b", "c", "d"}),
            ],
            sample,
        )

    def test_sample_zero_output_convex_subgraphs_respects_alternate_graph(self) -> None:
        graph = nx.DiGraph()
        graph.add_node("p", forbidden=True)
        graph.add_edges_from(
            [
                ("p", "a"),
                ("p", "b"),
                ("p", "c"),
                ("a", "c"),
            ]
        )
        alternate_graph = nx.DiGraph()
        alternate_graph.add_nodes_from(graph.nodes(data=True))
        alternate_graph.add_edges_from(
            [
                ("a", "b"),
                ("b", "c"),
            ]
        )

        result = [
            frozenset(nodes)
            for nodes in sample_zero_output_convex_subgraphs(
                graph,
                1,
                alternate_graph=alternate_graph,
                forbid_sources_and_sinks=False,
                max_states_expanded=64,
                max_samples=8,
                max_children_per_state=2,
                size_bin_width=1,
                **BODY_ONLY_FORBIDDEN,
            )
        ]

        self.assertEqual(len(result), len(set(result)))
        self.assertIn(frozenset({"a", "b", "c"}), result)
        self.assertNotIn(frozenset({"a", "c"}), result)

    def test_grow_zero_output_convex_subgraphs_enumerates_supersets(self) -> None:
        graph = nx.DiGraph()
        graph.add_node("p", forbidden=True)
        graph.add_edges_from(
            [
                ("p", "a"),
                ("p", "b"),
                ("p", "c"),
            ]
        )

        result = {
            frozenset(nodes)
            for nodes in grow_zero_output_convex_subgraphs(
                graph,
                None,
                seed_nodes={"a"},
                max_num_inputs=1,
                forbid_sources_and_sinks=False,
                **BODY_ONLY_FORBIDDEN,
            )
        }

        self.assertSetEqual(
            {
                frozenset({"a"}),
                frozenset({"a", "b"}),
                frozenset({"a", "c"}),
                frozenset({"a", "b", "c"}),
            },
            result,
        )

    def test_grow_zero_output_convex_subgraphs_oracle_prunes_growth(self) -> None:
        graph = nx.DiGraph()
        graph.add_node("p", forbidden=True)
        graph.add_edges_from(
            [
                ("p", "a"),
                ("p", "b"),
                ("p", "c"),
            ]
        )

        result = {
            frozenset(nodes)
            for nodes in grow_zero_output_convex_subgraphs(
                graph,
                None,
                seed_nodes={"a"},
                max_num_inputs=1,
                forbid_sources_and_sinks=False,
                oracle=lambda _state, nodes: True if len(nodes) < 2 else None,
                initial_oracle_state=True,
                **BODY_ONLY_FORBIDDEN,
            )
        }

        self.assertSetEqual(
            {
                frozenset({"a"}),
                frozenset({"a", "b"}),
                frozenset({"a", "c"}),
            },
            result,
        )

    def test_grow_zero_output_convex_subgraphs_respects_alternate_graph(self) -> None:
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
        alternate_graph.add_nodes_from(graph.nodes(data=True))
        alternate_graph.add_edges_from(
            [
                ("a", "b"),
                ("b", "c"),
            ]
        )

        result = {
            frozenset(nodes)
            for nodes in grow_zero_output_convex_subgraphs(
                graph,
                alternate_graph,
                seed_nodes={"a"},
                max_num_inputs=1,
                forbid_sources_and_sinks=False,
                **BODY_ONLY_FORBIDDEN,
            )
        }

        self.assertSetEqual(
            {
                frozenset({"a"}),
                frozenset({"a", "b"}),
                frozenset({"a", "b", "c"}),
            },
            result,
        )

    def test_grow_zero_output_convex_subgraphs_threads_oracle_state(self) -> None:
        graph = nx.DiGraph()
        graph.add_node("p", forbidden=True)
        graph.add_edges_from(
            [
                ("p", "a"),
                ("p", "b"),
                ("p", "c"),
            ]
        )

        seen_states: dict[frozenset[Hashable], object] = {}

        def oracle(state: object, nodes: set[Hashable]) -> object | None:
            seen_states[frozenset(nodes)] = state
            return frozenset(nodes)

        result = {
            frozenset(nodes)
            for nodes in grow_zero_output_convex_subgraphs(
                graph,
                None,
                seed_nodes={"a"},
                max_num_inputs=1,
                forbid_sources_and_sinks=False,
                oracle=oracle,
                initial_oracle_state="seed-state",
                **BODY_ONLY_FORBIDDEN,
            )
        }

        self.assertSetEqual(
            {
                frozenset({"a"}),
                frozenset({"a", "b"}),
                frozenset({"a", "c"}),
                frozenset({"a", "b", "c"}),
            },
            result,
        )
        self.assertEqual("seed-state", seen_states[frozenset({"a"})])
        self.assertEqual(
            frozenset({"a"}),
            seen_states[frozenset({"a", "b"})],
        )
        self.assertEqual(
            frozenset({"a"}),
            seen_states[frozenset({"a", "c"})],
        )

    def test_grow_zero_output_convex_subgraphs_merged_path_uses_deterministic_parent_state(
        self,
    ) -> None:
        graph = nx.DiGraph()
        graph.add_node("p", forbidden=True)
        graph.add_edges_from(
            [
                ("p", "a"),
                ("p", "b"),
                ("p", "c"),
            ]
        )

        seen_states: dict[frozenset[Hashable], object] = {}

        def oracle(state: object, nodes: set[Hashable]) -> object | None:
            seen_states[frozenset(nodes)] = state
            return frozenset(nodes)

        list(
            grow_zero_output_convex_subgraphs(
                graph,
                None,
                seed_nodes={"a"},
                max_num_inputs=1,
                forbid_sources_and_sinks=False,
                oracle=oracle,
                initial_oracle_state="seed-state",
                **BODY_ONLY_FORBIDDEN,
            )
        )

        self.assertEqual(
            frozenset({"a", "c"}),
            seen_states[frozenset({"a", "b", "c"})],
        )

    def test_zero_output_role_reversal_respects_original_input_bound(self) -> None:
        graph = nx.DiGraph()
        graph.add_edges_from(
            [
                ("i1", "a"),
                ("i2", "a"),
                ("i3", "b"),
                ("i4", "b"),
                ("i5", "c"),
                ("a", "b"),
                ("b", "c"),
            ]
        )

        result = {
            frozenset(nodes)
            for nodes in enumerate_convex_subgraphs(
                graph,
                4,
                1,
                forbid_sources_and_sinks=False,
                allow_zero_outputs=True,
            )
        }

        self.assertNotIn(frozenset({"a", "b", "c"}), result)
        self.assertIn(frozenset({"b", "c"}), result)

    def test_zero_output_role_reversal_maximum_respects_original_input_bound(
        self,
    ) -> None:
        graph = nx.DiGraph()
        graph.add_node("a", weight=5.0)
        graph.add_node("b", weight=5.0)
        graph.add_node("c", weight=5.0)
        graph.add_node("i1", weight=1.0, forbidden=True)
        graph.add_node("i2", weight=1.0, forbidden=True)
        graph.add_node("i3", weight=1.0, forbidden=True)
        graph.add_node("i4", weight=1.0, forbidden=True)
        graph.add_node("i5", weight=1.0, forbidden=True)
        graph.add_edges_from(
            [
                ("i1", "a"),
                ("i2", "a"),
                ("i3", "b"),
                ("i4", "b"),
                ("i5", "c"),
                ("a", "b"),
                ("b", "c"),
            ]
        )

        result = {
            frozenset(nodes)
            for nodes in enumerate_maximum_convex_subgraphs(
                graph,
                4,
                0,
                weighted=True,
                forbid_sources_and_sinks=False,
                allow_zero_outputs=True,
                **BODY_ONLY_FORBIDDEN,
            )
        }

        self.assertSetEqual({frozenset({"b", "c"})}, result)

    def test_zero_output_enumeration_allows_shared_original_inputs(self) -> None:
        graph = nx.DiGraph()
        graph.add_node("i", forbidden=True)
        graph.add_edges_from(
            [
                ("i", "a"),
                ("i", "b"),
                ("i", "c"),
                ("a", "b"),
                ("b", "c"),
            ]
        )

        result = {
            frozenset(nodes)
            for nodes in enumerate_convex_subgraphs(
                graph,
                1,
                1,
                forbid_sources_and_sinks=False,
                allow_zero_outputs=True,
                **BODY_ONLY_FORBIDDEN,
            )
        }

        self.assertIn(frozenset({"a", "b", "c"}), result)

    def test_connected_only_zero_output_connects_through_inputs(
        self,
    ) -> None:
        graph = nx.DiGraph()
        graph.add_node("p", forbidden=True)
        graph.add_edges_from(
            [
                ("p", "a"),
                ("p", "b"),
            ]
        )

        unconstrained = {
            frozenset(nodes)
            for nodes in enumerate_convex_subgraphs(
                graph,
                1,
                1,
                forbid_sources_and_sinks=False,
                allow_zero_outputs=True,
                **BODY_ONLY_FORBIDDEN,
            )
        }
        connected = {
            frozenset(nodes)
            for nodes in enumerate_convex_subgraphs(
                graph,
                1,
                1,
                forbid_sources_and_sinks=False,
                allow_zero_outputs=True,
                connected_only=True,
                **BODY_ONLY_FORBIDDEN,
            )
        }

        self.assertIn(frozenset({"a", "b"}), unconstrained)
        self.assertIn(frozenset({"a", "b"}), connected)
        self.assertIn(frozenset({"a"}), connected)
        self.assertIn(frozenset({"b"}), connected)

    def test_connected_only_zero_output_maximum_connects_through_inputs(
        self,
    ) -> None:
        graph = nx.DiGraph()
        graph.add_node("p", forbidden=True)
        graph.add_node("a", weight=3.0)
        graph.add_node("b", weight=3.0)
        graph.add_edges_from(
            [
                ("p", "a"),
                ("p", "b"),
            ]
        )

        unconstrained = {
            frozenset(nodes)
            for nodes in enumerate_maximum_convex_subgraphs(
                graph,
                1,
                1,
                weighted=True,
                forbid_sources_and_sinks=False,
                allow_zero_outputs=True,
                **BODY_ONLY_FORBIDDEN,
            )
        }
        connected = {
            frozenset(nodes)
            for nodes in enumerate_maximum_convex_subgraphs(
                graph,
                1,
                1,
                weighted=True,
                forbid_sources_and_sinks=False,
                allow_zero_outputs=True,
                connected_only=True,
                **BODY_ONLY_FORBIDDEN,
            )
        }

        self.assertSetEqual({frozenset({"a", "b"})}, unconstrained)
        self.assertSetEqual({frozenset({"a", "b"})}, connected)

    def test_connected_only_zero_output_does_not_use_input_input_edges(
        self,
    ) -> None:
        graph = nx.DiGraph()
        graph.add_node("a", forbidden=True)
        graph.add_node("d", forbidden=True)
        graph.add_edges_from(
            [
                ("a", "b"),
                ("d", "c"),
                ("a", "d"),
            ]
        )

        unconstrained = {
            frozenset(nodes)
            for nodes in enumerate_convex_subgraphs(
                graph,
                2,
                1,
                forbid_sources_and_sinks=False,
                allow_zero_outputs=True,
                **BODY_ONLY_FORBIDDEN,
            )
        }
        connected = {
            frozenset(nodes)
            for nodes in enumerate_convex_subgraphs(
                graph,
                2,
                1,
                forbid_sources_and_sinks=False,
                allow_zero_outputs=True,
                connected_only=True,
                **BODY_ONLY_FORBIDDEN,
            )
        }

        self.assertIn(frozenset({"b", "c"}), unconstrained)
        self.assertNotIn(frozenset({"b", "c"}), connected)

    def test_alternate_connected_only_filters_independently(self) -> None:
        graph = nx.DiGraph()
        graph.add_edges_from([("a", "b")])

        alternate_graph = nx.DiGraph()
        alternate_graph.add_nodes_from(["a", "b"])

        primary_connected = {
            frozenset(nodes)
            for nodes in enumerate_convex_subgraphs(
                graph,
                0,
                0,
                alternate_graph=alternate_graph,
                forbid_sources_and_sinks=False,
                connected_only=True,
                sampling=False,
            )
        }
        alternate_connected = {
            frozenset(nodes)
            for nodes in enumerate_convex_subgraphs(
                graph,
                0,
                0,
                alternate_graph=alternate_graph,
                forbid_sources_and_sinks=False,
                connected_only=True,
                alternate_connected_only=True,
                sampling=False,
            )
        }

        self.assertIn(frozenset({"a", "b"}), primary_connected)
        self.assertNotIn(frozenset({"a", "b"}), alternate_connected)

    def test_alternate_connected_only_uses_unbounded_alternate_inputs(self) -> None:
        graph = nx.DiGraph()
        graph.add_edges_from([("a", "b")])

        alternate_graph = nx.DiGraph()
        alternate_graph.add_node("x", forbidden=True)
        alternate_graph.add_edges_from(
            [
                ("x", "a"),
                ("x", "b"),
            ]
        )

        alternate_connected = {
            frozenset(nodes)
            for nodes in enumerate_convex_subgraphs(
                graph,
                0,
                0,
                alternate_graph=alternate_graph,
                forbid_sources_and_sinks=False,
                alternate_connected_only=True,
                sampling=False,
                **BODY_ONLY_FORBIDDEN,
            )
        }

        self.assertIn(frozenset({"a", "b"}), alternate_connected)

    def test_alternate_connected_only_does_not_require_primary_connectivity(self) -> None:
        graph = nx.DiGraph()
        graph.add_nodes_from(["a", "b"])

        alternate_graph = nx.DiGraph()
        alternate_graph.add_node("x", forbidden=True)
        alternate_graph.add_edges_from(
            [
                ("x", "a"),
                ("x", "b"),
            ]
        )

        alternate_connected = {
            frozenset(nodes)
            for nodes in enumerate_convex_subgraphs(
                graph,
                0,
                0,
                alternate_graph=alternate_graph,
                forbid_sources_and_sinks=False,
                alternate_connected_only=True,
                sampling=False,
                **BODY_ONLY_FORBIDDEN,
            )
        }
        primary_connected = {
            frozenset(nodes)
            for nodes in enumerate_convex_subgraphs(
                graph,
                0,
                0,
                alternate_graph=alternate_graph,
                forbid_sources_and_sinks=False,
                connected_only=True,
                sampling=False,
                **BODY_ONLY_FORBIDDEN,
            )
        }

        self.assertIn(frozenset({"a", "b"}), alternate_connected)
        self.assertNotIn(frozenset({"a", "b"}), primary_connected)

    def test_alternate_input_forbidden_does_not_restrict_primary_inputs(self) -> None:
        graph = nx.DiGraph()
        graph.add_edges_from([("i", "a")])

        alternate_graph = nx.DiGraph()
        alternate_graph.add_node("i", input_forbidden=True)
        alternate_graph.add_node("a")
        alternate_graph.add_edge("i", "a")

        result = {
            frozenset(nodes)
            for nodes in enumerate_convex_subgraphs(
                graph,
                1,
                0,
                alternate_graph=alternate_graph,
                forbid_sources_and_sinks=False,
                allow_zero_outputs=True,
                sampling=False,
                forbidden_attr=None,
                input_forbidden_attr="input_forbidden",
            )
        }

        self.assertIn(frozenset({"a"}), result)

    def test_alternate_outputs_do_not_count_against_primary_output_limit(self) -> None:
        graph = nx.DiGraph()
        graph.add_nodes_from(["a", "b", "c", "i1", "i2", "out"])
        graph.add_edges_from(
            [
                ("i1", "c"),
                ("i2", "c"),
                ("c", "out"),
            ]
        )

        alternate_graph = nx.DiGraph()
        alternate_graph.add_nodes_from(graph.nodes)
        alternate_graph.add_node("ai", forbidden=True)
        alternate_graph.add_node("ao1", forbidden=True)
        alternate_graph.add_node("ao2", forbidden=True)
        alternate_graph.add_node("ao3", forbidden=True)
        alternate_graph.add_edges_from(
            [
                ("ai", "a"),
                ("ai", "b"),
                ("ai", "c"),
                ("a", "ao1"),
                ("b", "ao2"),
                ("c", "ao3"),
            ]
        )

        result = {
            frozenset(nodes)
            for nodes in enumerate_convex_subgraphs(
                graph,
                2,
                1,
                alternate_graph=alternate_graph,
                forbid_sources_and_sinks=False,
                alternate_connected_only=True,
                sampling=False,
                **BODY_ONLY_FORBIDDEN,
            )
        }

        self.assertIn(frozenset({"a", "b", "c"}), result)

    def test_alternate_connected_only_completes_successors_of_primary_output(
        self,
    ) -> None:
        graph = nx.DiGraph()
        graph.add_node("x", body_forbidden=True, input_forbidden=True)
        graph.add_edges_from(
            [
                ("a", "b"),
                ("a", "c"),
                ("a", "x"),
                ("d", "e"),
                ("e", "c"),
            ]
        )
        alternate_graph = graph.copy()

        result = {
            frozenset(nodes)
            for nodes in enumerate_convex_subgraphs(
                graph,
                0,
                1,
                alternate_graph=alternate_graph,
                forbid_sources_and_sinks=False,
                alternate_connected_only=True,
                sampling=False,
                forbidden_attr=None,
                body_forbidden_attr="body_forbidden",
                input_forbidden_attr="input_forbidden",
            )
        }

        self.assertIn(frozenset({"a", "b", "c", "d", "e"}), result)

    def test_alternate_connected_only_requires_alternate_graph(self) -> None:
        graph = nx.DiGraph()
        graph.add_edges_from([("a", "b")])

        with self.assertRaisesRegex(ValueError, "alternate_connected_only"):
            list(
                enumerate_convex_subgraphs(
                    graph,
                    0,
                    0,
                    forbid_sources_and_sinks=False,
                    alternate_connected_only=True,
                    sampling=False,
                )
            )

    def test_alternate_graph_filters_zero_output_results(self) -> None:
        graph = nx.DiGraph()
        graph.add_node("p", forbidden=True)
        graph.add_nodes_from(["a", "b", "c"])
        graph.add_edges_from(
            [
                ("p", "a"),
                ("p", "c"),
            ]
        )

        alternate_graph = nx.DiGraph()
        alternate_graph.add_nodes_from(["a", "b", "c"])
        alternate_graph.add_edges_from(
            [
                ("a", "b"),
                ("b", "c"),
            ]
        )

        without_alternate = {
            frozenset(nodes)
            for nodes in enumerate_convex_subgraphs(
                graph,
                1,
                1,
                forbid_sources_and_sinks=False,
                allow_zero_outputs=True,
                **BODY_ONLY_FORBIDDEN,
            )
        }
        with_alternate = {
            frozenset(nodes)
            for nodes in enumerate_convex_subgraphs(
                graph,
                1,
                1,
                alternate_graph=alternate_graph,
                forbid_sources_and_sinks=False,
                allow_zero_outputs=True,
                **BODY_ONLY_FORBIDDEN,
            )
        }

        self.assertIn(frozenset({"a", "c"}), without_alternate)
        self.assertNotIn(frozenset({"a", "c"}), with_alternate)
        self.assertIn(frozenset({"a", "b", "c"}), with_alternate)

    def test_maximum_enumeration_respects_alternate_graph(self) -> None:
        graph = nx.DiGraph()
        graph.add_node("p", forbidden=True, weight=0.0)
        graph.add_node("a", weight=10.0)
        graph.add_node("b", weight=1.0)
        graph.add_node("c", weight=10.0)
        graph.add_edges_from(
            [
                ("p", "a"),
                ("p", "c"),
            ]
        )

        alternate_graph = nx.DiGraph()
        alternate_graph.add_nodes_from(graph.nodes(data=True))
        alternate_graph.add_edges_from(
            [
                ("a", "b"),
                ("b", "c"),
            ]
        )

        kwargs = {
            "max_num_inputs": 1,
            "max_num_outputs": 1,
            "max_subgraph_size": 2,
            "weighted": True,
            "forbid_sources_and_sinks": False,
            "allow_zero_outputs": True,
            **BODY_ONLY_FORBIDDEN,
        }
        without_alternate = {
            frozenset(nodes)
            for nodes in enumerate_maximum_convex_subgraphs(graph, **kwargs)
        }
        with_alternate = {
            frozenset(nodes)
            for nodes in enumerate_maximum_convex_subgraphs(
                graph,
                alternate_graph=alternate_graph,
                **kwargs,
            )
        }

        self.assertSetEqual({frozenset({"a", "c"})}, without_alternate)
        self.assertSetEqual(
            {frozenset({"a", "b"}), frozenset({"b", "c"})},
            with_alternate,
        )

    def test_sampled_single_output_maximum_can_include_zero_outputs(self) -> None:
        graph = nx.DiGraph()
        graph.add_node("p", forbidden=True)
        graph.add_nodes_from(["a", "b"])
        graph.add_edges_from(
            [
                ("p", "a"),
                ("p", "b"),
            ]
        )

        alternate_graph = nx.DiGraph()
        alternate_graph.add_nodes_from(graph.nodes(data=True))

        result = {
            frozenset(nodes)
            for nodes in enumerate_maximum_convex_subgraphs(
                graph,
                1,
                1,
                alternate_graph=alternate_graph,
                forbid_sources_and_sinks=False,
                allow_zero_outputs=True,
                sampling=True,
                **BODY_ONLY_FORBIDDEN,
            )
        }

        self.assertSetEqual({frozenset({"a", "b"})}, result)

    def test_auto_sampling_discards_buffered_exhaustive_results_after_threshold(
        self,
    ) -> None:
        graph = nx.DiGraph()
        graph.add_node("src", forbidden=True)
        graph.add_node("sink", forbidden=True)
        graph.add_edges_from(
            [
                ("src", "a"),
                ("a", "b"),
                ("b", "sink"),
            ]
        )

        with (
            patch.object(mvs_api, "_AUTO_SAMPLING_RESULT_THRESHOLD", 2),
            patch.object(
                mvs_api._ConvexSubgraphOperation,
                "_iter_sampled_convex_subgraphs",
                return_value=iter([{"sample"}]),
            ) as sampled,
        ):
            result = list(
                enumerate_convex_subgraphs(
                    graph,
                    1,
                    1,
                    **BODY_ONLY_FORBIDDEN,
                )
            )

        self.assertEqual([{"sample"}], result)
        sampled.assert_called_once()

    def test_sampling_false_keeps_exhaustive_results_after_threshold(
        self,
    ) -> None:
        graph = nx.DiGraph()
        graph.add_node("src", forbidden=True)
        graph.add_node("sink", forbidden=True)
        graph.add_edges_from(
            [
                ("src", "a"),
                ("a", "b"),
                ("b", "sink"),
            ]
        )

        with (
            patch.object(mvs_api, "_AUTO_SAMPLING_RESULT_THRESHOLD", 1),
            patch.object(
                mvs_api._ConvexSubgraphOperation,
                "_iter_sampled_convex_subgraphs",
                return_value=iter([{"sample"}]),
            ) as sampled,
        ):
            result = {
                frozenset(nodes)
                for nodes in enumerate_convex_subgraphs(
                    graph,
                    1,
                    1,
                    sampling=False,
                    **BODY_ONLY_FORBIDDEN,
                )
            }

        self.assertSetEqual(
            {
                frozenset({"a"}),
                frozenset({"b"}),
                frozenset({"a", "b"}),
            },
            result,
        )
        sampled.assert_not_called()

    def test_sampling_true_uses_sampler_immediately(self) -> None:
        graph = nx.DiGraph()
        graph.add_node("src", forbidden=True)
        graph.add_edge("src", "a")

        with patch.object(
            mvs_api._ConvexSubgraphOperation,
            "_iter_sampled_convex_subgraphs",
            return_value=iter([{"sample"}]),
        ) as sampled:
            result = list(
                enumerate_convex_subgraphs(
                    graph,
                    1,
                    1,
                    sampling=True,
                    **BODY_ONLY_FORBIDDEN,
                )
            )

        self.assertEqual([{"sample"}], result)
        sampled.assert_called_once()

    def test_maximum_auto_sampling_uses_sampled_candidates_after_threshold(
        self,
    ) -> None:
        graph = nx.DiGraph()
        graph.add_node("src", forbidden=True)
        graph.add_node("sink", forbidden=True)
        graph.add_edges_from(
            [
                ("src", "a"),
                ("a", "b"),
                ("b", "sink"),
            ]
        )

        with (
            patch.object(mvs_api, "_AUTO_SAMPLING_RESULT_THRESHOLD", 1),
            patch.object(
                mvs_api._ConvexSubgraphOperation,
                "_iter_sampled_convex_subgraphs",
                return_value=iter([{"a"}]),
            ) as sampled,
        ):
            result = {
                frozenset(nodes)
                for nodes in enumerate_maximum_convex_subgraphs(
                    graph,
                    1,
                    1,
                    **BODY_ONLY_FORBIDDEN,
                )
            }

        self.assertSetEqual({frozenset({"a"})}, result)
        sampled.assert_called_once()

    def test_alternate_graph_closure_rejects_body_forbidden_nodes(self) -> None:
        graph = nx.DiGraph()
        graph.add_node("p", forbidden=True, weight=0.0)
        graph.add_node("a", weight=10.0)
        graph.add_node("b", forbidden=True, weight=1.0)
        graph.add_node("c", weight=10.0)
        graph.add_edges_from(
            [
                ("p", "a"),
                ("p", "c"),
            ]
        )

        alternate_graph = nx.DiGraph()
        alternate_graph.add_nodes_from(graph.nodes(data=True))
        alternate_graph.add_edges_from(
            [
                ("a", "b"),
                ("b", "c"),
            ]
        )

        result = {
            frozenset(nodes)
            for nodes in enumerate_maximum_convex_subgraphs(
                graph,
                1,
                1,
                alternate_graph=alternate_graph,
                weighted=True,
                forbid_sources_and_sinks=False,
                allow_zero_outputs=True,
                **BODY_ONLY_FORBIDDEN,
            )
        }

        self.assertSetEqual({frozenset({"a"}), frozenset({"c"})}, result)

    def test_alternate_graph_filters_connected_zero_output_results(self) -> None:
        graph = nx.DiGraph()
        graph.add_node("p", forbidden=True)
        graph.add_nodes_from(["a", "b", "c"])
        graph.add_edges_from(
            [
                ("p", "a"),
                ("p", "b"),
                ("p", "c"),
            ]
        )

        alternate_graph = nx.DiGraph()
        alternate_graph.add_nodes_from(["a", "b", "c"])
        alternate_graph.add_edges_from(
            [
                ("a", "b"),
                ("b", "c"),
            ]
        )

        connected = {
            frozenset(nodes)
            for nodes in enumerate_convex_subgraphs(
                graph,
                1,
                1,
                forbid_sources_and_sinks=False,
                allow_zero_outputs=True,
                connected_only=True,
                **BODY_ONLY_FORBIDDEN,
            )
        }
        connected_with_alternate = {
            frozenset(nodes)
            for nodes in enumerate_convex_subgraphs(
                graph,
                1,
                1,
                alternate_graph=alternate_graph,
                forbid_sources_and_sinks=False,
                allow_zero_outputs=True,
                connected_only=True,
                **BODY_ONLY_FORBIDDEN,
            )
        }

        self.assertIn(frozenset({"a", "c"}), connected)
        self.assertNotIn(frozenset({"a", "c"}), connected_with_alternate)
        self.assertIn(frozenset({"a", "b", "c"}), connected_with_alternate)

    def test_direct_zero_output_connected_with_alternate_graph(self) -> None:
        graph = nx.DiGraph()
        graph.add_node("p", forbidden=True)
        graph.add_nodes_from(["a", "b", "c"])
        graph.add_edges_from(
            [
                ("p", "a"),
                ("p", "b"),
                ("p", "c"),
            ]
        )

        alternate_graph = nx.DiGraph()
        alternate_graph.add_nodes_from(["a", "b", "c"])
        alternate_graph.add_edges_from(
            [
                ("a", "b"),
                ("b", "c"),
            ]
        )

        result = {
            frozenset(nodes)
            for nodes in enumerate_convex_subgraphs(
                graph,
                1,
                0,
                alternate_graph=alternate_graph,
                forbid_sources_and_sinks=False,
                connected_only=True,
                **BODY_ONLY_FORBIDDEN,
            )
        }

        self.assertSetEqual(
            {
                frozenset({"a"}),
                frozenset({"b"}),
                frozenset({"c"}),
                frozenset({"a", "b"}),
                frozenset({"b", "c"}),
                frozenset({"a", "b", "c"}),
            },
            result,
        )

    def test_direct_zero_output_connected_with_alternate_graph_has_no_duplicates(
        self,
    ) -> None:
        graph = nx.DiGraph()
        graph.add_node("p", forbidden=True)
        graph.add_nodes_from(["a", "b", "c", "d"])
        graph.add_edges_from(
            [
                ("p", "a"),
                ("p", "b"),
                ("p", "c"),
                ("p", "d"),
            ]
        )

        alternate_graph = nx.DiGraph()
        alternate_graph.add_nodes_from(["a", "b", "c", "d"])
        alternate_graph.add_edges_from(
            [
                ("a", "c"),
                ("b", "c"),
                ("c", "d"),
            ]
        )

        result = list(
            enumerate_convex_subgraphs(
                graph,
                1,
                0,
                alternate_graph=alternate_graph,
                forbid_sources_and_sinks=False,
                connected_only=True,
                **BODY_ONLY_FORBIDDEN,
            )
        )
        unique = {frozenset(nodes) for nodes in result}

        self.assertEqual(len(result), len(unique))

    def test_direct_zero_output_connected_with_alternate_graph_respects_output_bound(
        self,
    ) -> None:
        graph = nx.DiGraph()
        graph.add_node("p", forbidden=True)
        graph.add_nodes_from(["a", "sink"])
        graph.add_edges_from(
            [
                ("p", "a"),
                ("a", "sink"),
            ]
        )

        alternate_graph = nx.DiGraph()
        alternate_graph.add_nodes_from(["a", "sink"])
        alternate_graph.add_edge("a", "sink")

        result = {
            frozenset(nodes)
            for nodes in enumerate_convex_subgraphs(
                graph,
                1,
                0,
                alternate_graph=alternate_graph,
                forbid_sources_and_sinks=False,
                connected_only=True,
                **BODY_ONLY_FORBIDDEN,
            )
        }

        self.assertNotIn(frozenset({"a"}), result)
        self.assertIn(frozenset({"sink"}), result)
        self.assertIn(frozenset({"a", "sink"}), result)

    def test_alternate_graph_allows_extra_forbidden_nodes(self) -> None:
        graph = nx.DiGraph()
        graph.add_nodes_from(["a", "b"])
        graph.add_edge("a", "b")

        alternate_graph = nx.DiGraph()
        alternate_graph.add_node("p", forbidden=True)
        alternate_graph.add_nodes_from(["a", "b"])
        alternate_graph.add_edge("a", "b")

        without_alternate = {
            frozenset(nodes)
            for nodes in enumerate_convex_subgraphs(
                graph,
                1,
                1,
                forbid_sources_and_sinks=False,
            )
        }
        with_alternate = {
            frozenset(nodes)
            for nodes in enumerate_convex_subgraphs(
                graph,
                1,
                1,
                alternate_graph=alternate_graph,
                forbid_sources_and_sinks=False,
            )
        }

        self.assertSetEqual(without_alternate, with_alternate)

    def test_alternate_graph_rejects_missing_non_forbidden_nodes(self) -> None:
        graph = nx.DiGraph()
        graph.add_nodes_from(["a", "b"])
        graph.add_edge("a", "b")

        alternate_graph = nx.DiGraph()
        alternate_graph.add_node("a")

        with self.assertRaisesRegex(
            ValueError,
            "alternate_graph may only omit nodes that are forbidden in graph",
        ):
            graph_to_input(graph, alternate_graph=alternate_graph)

    def test_connected_only_alternate_graph_respects_cross_component_paths(self) -> None:
        graph = nx.DiGraph()
        graph.add_node("src", forbidden=True)
        graph.add_nodes_from(["a", "b", "x"])
        graph.add_edges_from(
            [
                ("src", "a"),
                ("a", "b"),
            ]
        )

        alternate_graph = nx.DiGraph()
        alternate_graph.add_nodes_from(graph.nodes(data=True))
        alternate_graph.add_edges_from(
            [
                ("a", "x"),
                ("x", "b"),
            ]
        )

        result = {
            frozenset(nodes)
            for nodes in enumerate_convex_subgraphs(
                graph,
                1,
                1,
                alternate_graph=alternate_graph,
                forbid_sources_and_sinks=False,
                connected_only=True,
                **BODY_ONLY_FORBIDDEN,
            )
        }

        self.assertNotIn(frozenset({"a", "b"}), result)

    def test_string_forbidden_attributes_are_parsed_as_booleans(self) -> None:
        graph = nx.DiGraph()
        graph.add_node("src", forbidden="True")
        graph.add_node("mid", forbidden="False")
        graph.add_node("sink", forbidden="False")
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
                **BODY_ONLY_FORBIDDEN,
            )
        }

        self.assertIn(frozenset({"mid"}), result)

    def test_alternate_graph_not_supported_for_native_maximum_search(self) -> None:
        graph = nx.DiGraph()
        graph.add_node("p", forbidden=True)
        graph.add_nodes_from(["a", "b"])
        graph.add_edges_from(
            [
                ("p", "a"),
                ("p", "b"),
            ]
        )

        alternate_graph = nx.DiGraph()
        alternate_graph.add_nodes_from(graph.nodes(data=True))
        alternate_graph.add_edge("a", "b")

        with self.assertRaises(NotImplementedError):
            list(
                enumerate_maximum_convex_subgraphs(
                    graph,
                    1,
                    2,
                    alternate_graph=alternate_graph,
                    forbid_sources_and_sinks=False,
                )
            )

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

        result = {
            frozenset(nodes)
            for nodes in enumerate_convex_subgraphs(
                graph,
                1,
                1,
                **BODY_ONLY_FORBIDDEN,
            )
        }
        self.assertSetEqual(
            {
                frozenset({"a"}),
                frozenset({"b"}),
                frozenset({"a", "b"}),
            },
            result,
        )

    def test_exhaustive_enumeration_respects_max_subgraph_size(self) -> None:
        graph = nx.DiGraph()
        graph.add_node("src", forbidden=True)
        graph.add_edges_from(
            [
                ("src", "a"),
                ("a", "b"),
            ]
        )

        result = {
            frozenset(nodes)
            for nodes in enumerate_convex_subgraphs(
                graph,
                1,
                1,
                max_subgraph_size=1,
                forbid_sources_and_sinks=False,
                allow_zero_outputs=True,
                **BODY_ONLY_FORBIDDEN,
            )
        }

        self.assertSetEqual({frozenset({"a"}), frozenset({"b"})}, result)

    def test_connected_only_exhaustive_avoids_disconnected_unions(self) -> None:
        graph = nx.DiGraph()
        graph.add_node("src_a", forbidden=True)
        graph.add_node("src_b", forbidden=True)
        graph.add_edges_from(
            [
                ("src_a", "a"),
                ("src_b", "b"),
            ]
        )

        unconstrained = {
            frozenset(nodes)
            for nodes in enumerate_convex_subgraphs(
                graph,
                2,
                1,
                forbid_sources_and_sinks=False,
                allow_zero_outputs=True,
                **BODY_ONLY_FORBIDDEN,
            )
        }
        connected = {
            frozenset(nodes)
            for nodes in enumerate_convex_subgraphs(
                graph,
                2,
                1,
                forbid_sources_and_sinks=False,
                allow_zero_outputs=True,
                connected_only=True,
                **BODY_ONLY_FORBIDDEN,
            )
        }

        self.assertIn(frozenset({"a", "b"}), unconstrained)
        self.assertNotIn(frozenset({"a", "b"}), connected)
        self.assertIn(frozenset({"a"}), connected)
        self.assertIn(frozenset({"b"}), connected)

    def test_connected_zero_output_respects_max_subgraph_size(self) -> None:
        graph = nx.DiGraph()
        graph.add_node("p", forbidden=True)
        graph.add_edges_from(
            [
                ("p", "a"),
                ("p", "b"),
            ]
        )

        result = {
            frozenset(nodes)
            for nodes in enumerate_convex_subgraphs(
                graph,
                1,
                0,
                max_subgraph_size=1,
                forbid_sources_and_sinks=False,
                connected_only=True,
                **BODY_ONLY_FORBIDDEN,
            )
        }

        self.assertSetEqual({frozenset({"a"}), frozenset({"b"})}, result)

    def test_connected_only_maximum_avoids_disconnected_unions(self) -> None:
        graph = nx.DiGraph()
        graph.add_node("src_a", forbidden=True)
        graph.add_node("src_b", forbidden=True)
        graph.add_node("a", weight=3.0)
        graph.add_node("b", weight=2.0)
        graph.add_edges_from(
            [
                ("src_a", "a"),
                ("src_b", "b"),
            ]
        )

        unconstrained = {
            frozenset(nodes)
            for nodes in enumerate_maximum_convex_subgraphs(
                graph,
                2,
                1,
                weighted=True,
                forbid_sources_and_sinks=False,
                allow_zero_outputs=True,
                **BODY_ONLY_FORBIDDEN,
            )
        }
        connected = {
            frozenset(nodes)
            for nodes in enumerate_maximum_convex_subgraphs(
                graph,
                2,
                1,
                weighted=True,
                forbid_sources_and_sinks=False,
                allow_zero_outputs=True,
                connected_only=True,
                **BODY_ONLY_FORBIDDEN,
            )
        }

        self.assertSetEqual({frozenset({"a", "b"})}, unconstrained)
        self.assertSetEqual({frozenset({"a"})}, connected)

    def test_maximum_enumeration_respects_max_subgraph_size(self) -> None:
        graph = nx.DiGraph()
        graph.add_node("p", forbidden=True)
        graph.add_node("a", weight=3.0)
        graph.add_node("b", weight=3.0)
        graph.add_edges_from(
            [
                ("p", "a"),
                ("p", "b"),
            ]
        )

        result = {
            frozenset(nodes)
            for nodes in enumerate_maximum_convex_subgraphs(
                graph,
                1,
                2,
                max_subgraph_size=1,
                weighted=True,
                forbid_sources_and_sinks=False,
                **BODY_ONLY_FORBIDDEN,
            )
        }

        self.assertSetEqual({frozenset({"a"}), frozenset({"b"})}, result)

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
                        **BODY_ONLY_FORBIDDEN,
                    )
                )
                self.assertEqual(expected_count, len(result))


if __name__ == "__main__":
    unittest.main()
