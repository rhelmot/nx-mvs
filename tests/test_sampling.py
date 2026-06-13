from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
import pickle
import unittest
from typing import Literal

import networkx as nx

from mvs import (
    ConvexSubgraphQuery,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


def _sampling_query_kwargs(kwargs: dict[str, object]) -> dict[str, object]:
    sampling_names = {
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
    }
    return {sampling_names.get(name, name): value for name, value in kwargs.items()}


def enumerate_convex_subgraphs(
    graph: nx.DiGraph,
    max_num_inputs: int,
    max_num_outputs: int,
    **kwargs: object,
):
    sampling = kwargs.pop("sampling", None)
    max_queue_size = kwargs.pop("max_queue_size", 128)
    return ConvexSubgraphQuery(
        graph,
        max_num_inputs=max_num_inputs,
        max_num_outputs=max_num_outputs,
        **kwargs,
    ).enumerate(
        sampling=sampling,  # type: ignore[arg-type]
        max_queue_size=max_queue_size,  # type: ignore[arg-type]
    )


def sample_zero_output_convex_subgraphs(
    graph: nx.DiGraph,
    max_num_inputs: int,
    **kwargs: object,
):
    return ConvexSubgraphQuery(
        graph,
        max_num_inputs=max_num_inputs,
        max_num_outputs=0,
        **_sampling_query_kwargs(kwargs),
    ).sample()


def sample_nonzero_output_convex_subgraphs(
    graph: nx.DiGraph,
    max_num_inputs: int,
    max_num_outputs: int,
    **kwargs: object,
):
    return ConvexSubgraphQuery(
        graph,
        max_num_inputs=max_num_inputs,
        max_num_outputs=max_num_outputs,
        **_sampling_query_kwargs(kwargs),
    ).sample()


def grow_zero_output_convex_subgraphs(
    graph: nx.DiGraph,
    seed_nodes: set,
    **kwargs: object,
):
    oracle = kwargs.pop("oracle", None)
    initial_oracle_state = kwargs.pop("initial_oracle_state", None)
    return ConvexSubgraphQuery(
        graph,
        max_num_outputs=0,
        **kwargs,
    ).grow(
        seed_nodes,
        oracle=oracle,  # type: ignore[arg-type]
        initial_oracle_state=initial_oracle_state,
    )


def grow_nonzero_output_convex_subgraphs(
    graph: nx.DiGraph,
    seed_nodes: set,
    **kwargs: object,
):
    oracle = kwargs.pop("oracle", None)
    initial_oracle_state = kwargs.pop("initial_oracle_state", None)
    return ConvexSubgraphQuery(
        graph,
        **kwargs,
    ).grow(
        seed_nodes,
        oracle=oracle,  # type: ignore[arg-type]
        initial_oracle_state=initial_oracle_state,
    )


def _sample_sets(
    graph: nx.DiGraph[str],
    *,
    alternate_graph: nx.DiGraph[str] | None = None,
    max_num_inputs: int = 1,
    max_subgraph_size: int = 10,
    max_states_expanded: int = 8,
    max_samples: int = 32,
    max_children_per_state: int = 2,
    size_bin_width: int = 1,
    thicken_radius: int = 1,
    bucket_by_num_inputs: bool = True,
    minimal_node_bin_width: int = 1,
    ordering: Literal["default", "sort", "toposort"] = "toposort",
    sampling_passes: int = 1,
    exact_kernel_size: int = 0,
    forbidden_attr: str | None = "forbidden",
    body_forbidden_attr: str | None = None,
    input_forbidden_attr: str | None = None,
) -> set[frozenset[str]]:
    return {
        frozenset(nodes)
        for nodes in sample_zero_output_convex_subgraphs(
            graph,
            max_num_inputs,
            alternate_graph=alternate_graph,
            max_subgraph_size=max_subgraph_size,
            forbid_sources_and_sinks=False,
            max_states_expanded=max_states_expanded,
            max_samples=max_samples,
            max_children_per_state=max_children_per_state,
            size_bin_width=size_bin_width,
            thicken_radius=thicken_radius,
            bucket_by_num_inputs=bucket_by_num_inputs,
            minimal_node_bin_width=minimal_node_bin_width,
            ordering=ordering,
            sampling_passes=sampling_passes,
            exact_kernel_size=exact_kernel_size,
            forbidden_attr=forbidden_attr,
            body_forbidden_attr=body_forbidden_attr,
            input_forbidden_attr=input_forbidden_attr,
        )
    }


def _launch_distance(
    graph: nx.DiGraph[str],
    *,
    target: frozenset[str],
    samples: Iterable[frozenset[str]],
    alternate_graph: nx.DiGraph[str] | None = None,
    max_num_inputs: int = 1,
    max_subgraph_size: int = 10,
    forbidden_attr: str | None = "forbidden",
    body_forbidden_attr: str | None = None,
    input_forbidden_attr: str | None = None,
) -> int | None:
    best: int | None = None
    for sample in samples:
        if not sample.issubset(target):
            continue

        seen = {
            frozenset(nodes)
            for nodes in grow_zero_output_convex_subgraphs(
                graph,
                set(sample),
                alternate_graph=alternate_graph,
                max_num_inputs=max_num_inputs,
                max_subgraph_size=max_subgraph_size,
                forbid_sources_and_sinks=False,
                oracle=lambda _state, nodes, target=target: True if set(nodes).issubset(target) else None,
                initial_oracle_state=True,
                forbidden_attr=forbidden_attr,
                body_forbidden_attr=body_forbidden_attr,
                input_forbidden_attr=input_forbidden_attr,
            )
        }
        if target not in seen:
            continue
        distance = len(target) - len(sample)
        if best is None or distance < best:
            best = distance
    return best


def _read_pickle_graph(path: Path) -> nx.DiGraph[str]:
    with path.open("rb") as handle:
        return pickle.load(handle)


def _build_validator(
    graph: nx.DiGraph[str],
    alternate_graph: nx.DiGraph[str] | None = None,
):
    all_nodes = list(dict.fromkeys(list(graph.nodes()) + list(alternate_graph.nodes()) if alternate_graph is not None else list(graph.nodes())))
    node_index = {node: i for i, node in enumerate(all_nodes)}
    node_count = len(all_nodes)
    all_mask = (1 << node_count) - 1

    def as_bool(value: object) -> bool:
        if isinstance(value, str):
            return value.strip().lower() in {"1", "true", "t", "yes", "y", "on"}
        return bool(value)

    pred_main = [0] * node_count
    succ_main = [0] * node_count
    pred_alt = [0] * node_count
    succ_alt = [0] * node_count
    undirected_main = [0] * node_count
    body_forbidden = 0
    input_forbidden = 0

    for current_graph, pred_masks, succ_masks, is_main in (
        (graph, pred_main, succ_main, True),
        (alternate_graph, pred_alt, succ_alt, False),
    ):
        if current_graph is None:
            continue
        for node, attrs in current_graph.nodes(data=True):
            if as_bool(attrs.get("forbidden", False)):
                body_forbidden |= 1 << node_index[node]
                input_forbidden |= 1 << node_index[node]
            if as_bool(attrs.get("body_forbidden", False)):
                body_forbidden |= 1 << node_index[node]
            if as_bool(attrs.get("input_forbidden", False)):
                input_forbidden |= 1 << node_index[node]
        for source, target in current_graph.edges():
            u = node_index[source]
            v = node_index[target]
            succ_masks[u] |= 1 << v
            pred_masks[v] |= 1 << u
            if is_main:
                undirected_main[u] |= 1 << v
                undirected_main[v] |= 1 << u

    def topo_order(current_graph: nx.DiGraph[str] | None) -> list[int]:
        if current_graph is None:
            return []
        order = [node_index[node] for node in nx.topological_sort(current_graph)]
        present = set(current_graph.nodes())
        order.extend(node_index[node] for node in all_nodes if node not in present)
        return order

    def transitive_closure(
        pred_masks: list[int],
        succ_masks: list[int],
        topo: list[int],
    ) -> tuple[list[int], list[int]]:
        succ_tc = succ_masks[:]
        for u in reversed(topo):
            mask = succ_masks[u]
            while mask:
                lsb = mask & -mask
                v = lsb.bit_length() - 1
                succ_tc[u] |= succ_tc[v]
                mask ^= lsb
        pred_tc = pred_masks[:]
        for u in topo:
            mask = pred_masks[u]
            while mask:
                lsb = mask & -mask
                v = lsb.bit_length() - 1
                pred_tc[u] |= pred_tc[v]
                mask ^= lsb
        return pred_tc, succ_tc

    pred_tc_main, succ_tc_main = transitive_closure(pred_main, succ_main, topo_order(graph))
    if alternate_graph is not None:
        pred_tc_alt, succ_tc_alt = transitive_closure(pred_alt, succ_alt, topo_order(alternate_graph))
    else:
        pred_tc_alt, succ_tc_alt = None, None

    def mask_from_nodes(nodes: Iterable[str]) -> int:
        mask = 0
        for node in nodes:
            mask |= 1 << node_index[node]
        return mask

    def input_mask(mask: int) -> int:
        inputs = 0
        current = mask
        while current:
            lsb = current & -current
            u = lsb.bit_length() - 1
            inputs |= pred_main[u]
            current ^= lsb
        return inputs & ~mask

    def connected_with_inputs(mask: int) -> bool:
        inputs = input_mask(mask)
        augmented = mask | inputs
        if augmented == 0:
            return True
        start = (augmented & -augmented).bit_length() - 1
        seen = 1 << start
        stack = [start]
        while stack:
            u = stack.pop()
            neighbors = undirected_main[u] & augmented & ~seen
            while neighbors:
                lsb = neighbors & -neighbors
                v = lsb.bit_length() - 1
                if (inputs & (1 << u)) and (inputs & (1 << v)):
                    neighbors ^= lsb
                    continue
                seen |= 1 << v
                stack.append(v)
                neighbors ^= lsb
        return seen == augmented

    def is_convex(mask: int, pred_tc: list[int], succ_tc: list[int]) -> bool:
        outside = all_mask & ~mask
        while outside:
            lsb = outside & -outside
            w = lsb.bit_length() - 1
            if (pred_tc[w] & mask) and (succ_tc[w] & mask):
                return False
            outside ^= lsb
        return True

    def num_outputs(mask: int) -> int:
        outputs = 0
        current = mask
        while current:
            lsb = current & -current
            u = lsb.bit_length() - 1
            if succ_main[u] & ~mask:
                outputs += 1
            current ^= lsb
        return outputs

    def validate(nodes: Iterable[str]) -> None:
        mask = mask_from_nodes(nodes)
        assert (mask & body_forbidden) == 0
        assert (input_mask(mask) & input_forbidden) == 0
        assert input_mask(mask).bit_count() <= 4
        assert num_outputs(mask) == 0
        assert connected_with_inputs(mask)
        assert is_convex(mask, pred_tc_main, succ_tc_main)
        if pred_tc_alt is not None and succ_tc_alt is not None:
            assert is_convex(mask, pred_tc_alt, succ_tc_alt)

    return validate


class TestSampling(unittest.TestCase):
    def test_thickening_improves_low_budget_launch_coverage(self) -> None:
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

        thin = _sample_sets(
            graph,
            max_states_expanded=1,
            max_samples=8,
            max_children_per_state=2,
            size_bin_width=1,
            thicken_radius=0,
            forbidden_attr=None,
            body_forbidden_attr="forbidden",
        )
        thick = _sample_sets(
            graph,
            max_states_expanded=1,
            max_samples=8,
            max_children_per_state=2,
            size_bin_width=1,
            thicken_radius=2,
            forbidden_attr=None,
            body_forbidden_attr="forbidden",
        )

        self.assertTrue(thin < thick)
        target = frozenset({"a", "b", "c", "d"})
        self.assertNotIn(target, thin)
        self.assertIn(target, thick)
        self.assertEqual(
            2,
            _launch_distance(
                graph,
                target=target,
                samples=thin,
                forbidden_attr=None,
                body_forbidden_attr="forbidden",
            ),
        )
        self.assertEqual(
            0,
            _launch_distance(
                graph,
                target=target,
                samples=thick,
                forbidden_attr=None,
                body_forbidden_attr="forbidden",
            ),
        )

    def test_sampling_connected_zero_output_does_not_use_input_input_edges(
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

        samples = _sample_sets(
            graph,
            max_num_inputs=2,
            max_states_expanded=32,
            max_samples=32,
            max_children_per_state=4,
            size_bin_width=1,
            thicken_radius=1,
            forbidden_attr=None,
            body_forbidden_attr="forbidden",
        )

        self.assertNotIn(frozenset({"b", "c"}), samples)

    def test_extended_diversity_buckets_keep_distinct_families(self) -> None:
        graph = nx.DiGraph()
        graph.add_node("p", forbidden=True)
        graph.add_edges_from(
            [
                ("p", "d"),
                ("p", "e"),
                ("b", "e"),
                ("c", "d"),
                ("c", "e"),
            ]
        )

        exact = {
            frozenset(nodes)
            for nodes in enumerate_convex_subgraphs(
                graph,
                2,
                0,
                forbid_sources_and_sinks=False,
                connected_only=True,
                forbidden_attr=None,
                body_forbidden_attr="forbidden",
            )
        }
        self.assertIn(frozenset({"d"}), exact)
        self.assertIn(frozenset({"b", "d", "e"}), exact)

        size_only = _sample_sets(
            graph,
            max_num_inputs=2,
            max_states_expanded=3,
            max_samples=6,
            max_children_per_state=2,
            size_bin_width=10,
            thicken_radius=0,
            bucket_by_num_inputs=False,
            minimal_node_bin_width=0,
            forbidden_attr=None,
            body_forbidden_attr="forbidden",
        )
        extended = _sample_sets(
            graph,
            max_num_inputs=2,
            max_states_expanded=3,
            max_samples=6,
            max_children_per_state=2,
            size_bin_width=10,
            thicken_radius=0,
            bucket_by_num_inputs=True,
            minimal_node_bin_width=1,
            forbidden_attr=None,
            body_forbidden_attr="forbidden",
        )

        self.assertIn(frozenset({"d"}), extended)
        self.assertIn(frozenset({"b", "d", "e"}), extended)
        self.assertNotIn(frozenset({"d"}), size_only)
        self.assertNotIn(frozenset({"b", "d", "e"}), size_only)

    def test_multi_pass_sampling_improves_order_sensitive_coverage(self) -> None:
        graph = nx.DiGraph()
        graph.add_node("p", forbidden=True)
        graph.add_nodes_from(["d", "c", "b", "a"])
        graph.add_edges_from(
            [
                ("p", "d"),
                ("p", "c"),
                ("p", "b"),
                ("p", "a"),
            ]
        )

        single_pass = _sample_sets(
            graph,
            max_states_expanded=1,
            max_samples=4,
            max_children_per_state=1,
            size_bin_width=1,
            thicken_radius=0,
            ordering="default",
            sampling_passes=1,
            forbidden_attr=None,
            body_forbidden_attr="forbidden",
        )
        multi_pass = _sample_sets(
            graph,
            max_states_expanded=1,
            max_samples=4,
            max_children_per_state=1,
            size_bin_width=1,
            thicken_radius=0,
            ordering="default",
            sampling_passes=2,
            forbidden_attr=None,
            body_forbidden_attr="forbidden",
        )

        target = frozenset({"a", "b"})
        self.assertGreater(len(multi_pass), len(single_pass))
        self.assertIsNone(
            _launch_distance(
                graph,
                target=target,
                samples=single_pass,
                forbidden_attr=None,
                body_forbidden_attr="forbidden",
            )
        )
        self.assertEqual(
            0,
            _launch_distance(
                graph,
                target=target,
                samples=multi_pass,
                forbidden_attr=None,
                body_forbidden_attr="forbidden",
            ),
        )

    def test_exact_kernel_floor_restores_small_launch_points(self) -> None:
        graph = nx.DiGraph()
        graph.add_node("p", forbidden=True)
        graph.add_nodes_from(["d", "c", "b", "a"])
        graph.add_edges_from(
            [
                ("p", "d"),
                ("p", "c"),
                ("p", "b"),
                ("p", "a"),
            ]
        )

        heuristic_only = _sample_sets(
            graph,
            max_states_expanded=1,
            max_samples=4,
            max_children_per_state=1,
            size_bin_width=1,
            thicken_radius=0,
            ordering="default",
            sampling_passes=1,
            exact_kernel_size=0,
            forbidden_attr=None,
            body_forbidden_attr="forbidden",
        )
        with_kernels = _sample_sets(
            graph,
            max_states_expanded=1,
            max_samples=4,
            max_children_per_state=1,
            size_bin_width=1,
            thicken_radius=0,
            ordering="default",
            sampling_passes=1,
            exact_kernel_size=1,
            forbidden_attr=None,
            body_forbidden_attr="forbidden",
        )

        target = frozenset({"a", "b"})
        self.assertNotIn(frozenset({"a"}), heuristic_only)
        self.assertIn(frozenset({"a"}), with_kernels)
        self.assertIsNone(
            _launch_distance(
                graph,
                target=target,
                samples=heuristic_only,
                forbidden_attr=None,
                body_forbidden_attr="forbidden",
            )
        )
        self.assertEqual(
            1,
            _launch_distance(
                graph,
                target=target,
                samples=with_kernels,
                forbidden_attr=None,
                body_forbidden_attr="forbidden",
            ),
        )

    def test_real_graph_samples_are_unique_and_valid(self) -> None:
        if not (REPO_ROOT / "data/graph.p").exists() or not (REPO_ROOT / "data/graph-alt.p").exists():
            self.skipTest("repo-level graph pickle test inputs are not present")
        graph = _read_pickle_graph(REPO_ROOT / "data/graph.p")
        alternate_graph = _read_pickle_graph(REPO_ROOT / "data/graph-alt.p")
        validate = _build_validator(graph, alternate_graph)

        samples = list(
            sample_zero_output_convex_subgraphs(
                graph,
                4,
                alternate_graph=alternate_graph,
                max_subgraph_size=50,
                forbid_sources_and_sinks=False,
                max_states_expanded=128,
                max_samples=64,
                max_children_per_state=4,
                size_bin_width=2,
                thicken_radius=1,
                sampling_passes=2,
                exact_kernel_size=1,
            )
        )

        self.assertTrue(samples)
        self.assertEqual(len(samples), len({frozenset(sample) for sample in samples}))
        for sample in samples:
            validate(sample)

    def test_sampling_connected_nonzero_output_returns_valid_subset(self) -> None:
        graph = nx.DiGraph()
        graph.add_node("p", forbidden=True)
        graph.add_node("x", forbidden=True)
        graph.add_edges_from(
            [
                ("p", "a"),
                ("a", "b"),
                ("b", "x"),
            ]
        )

        exact = {
            frozenset(nodes)
            for nodes in enumerate_convex_subgraphs(
                graph,
                1,
                1,
                forbid_sources_and_sinks=False,
                connected_only=True,
                forbidden_attr=None,
                body_forbidden_attr="forbidden",
            )
        }
        samples = {
            frozenset(nodes)
            for nodes in sample_nonzero_output_convex_subgraphs(
                graph,
                1,
                1,
                forbid_sources_and_sinks=False,
                max_states_expanded=16,
                max_samples=8,
                max_children_per_state=2,
                size_bin_width=1,
                forbidden_attr=None,
                body_forbidden_attr="forbidden",
            )
        }

        self.assertSetEqual(
            {frozenset({"a"}), frozenset({"b"}), frozenset({"a", "b"})},
            samples,
        )
        self.assertTrue(samples <= exact)

    def test_grow_connected_nonzero_output_enumerates_valid_supersets(self) -> None:
        graph = nx.DiGraph()
        graph.add_node("p", forbidden=True)
        graph.add_node("x", forbidden=True)
        graph.add_edges_from(
            [
                ("p", "a"),
                ("a", "b"),
                ("b", "c"),
                ("c", "x"),
            ]
        )

        result = {
            frozenset(nodes)
            for nodes in grow_nonzero_output_convex_subgraphs(
                graph,
                {"a"},
                max_num_inputs=1,
                max_num_outputs=1,
                forbid_sources_and_sinks=False,
                forbidden_attr=None,
                body_forbidden_attr="forbidden",
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

    def test_grow_connected_nonzero_output_oracle_prunes_growth(self) -> None:
        graph = nx.DiGraph()
        graph.add_node("p", forbidden=True)
        graph.add_node("x", forbidden=True)
        graph.add_edges_from(
            [
                ("p", "a"),
                ("a", "b"),
                ("b", "c"),
                ("c", "x"),
            ]
        )

        result = {
            frozenset(nodes)
            for nodes in grow_nonzero_output_convex_subgraphs(
                graph,
                {"a"},
                max_num_inputs=1,
                max_num_outputs=1,
                forbid_sources_and_sinks=False,
                oracle=lambda _state, nodes: True if len(nodes) < 2 else None,
                initial_oracle_state=True,
                forbidden_attr=None,
                body_forbidden_attr="forbidden",
            )
        }

        self.assertSetEqual(
            {
                frozenset({"a"}),
                frozenset({"a", "b"}),
            },
            result,
        )

    def test_nonzero_output_sampling_and_growth_respect_alternate_graph(self) -> None:
        graph = nx.DiGraph()
        graph.add_node("p", forbidden=True)
        graph.add_node("x", forbidden=True)
        graph.add_edges_from(
            [
                ("p", "a"),
                ("p", "b"),
                ("a", "c"),
                ("b", "c"),
                ("c", "x"),
            ]
        )
        alternate_graph = nx.DiGraph()
        alternate_graph.add_nodes_from(graph.nodes(data=True))
        alternate_graph.add_edges_from(
            [
                ("a", "b"),
                ("b", "c"),
                ("c", "x"),
            ]
        )

        samples = {
            frozenset(nodes)
            for nodes in sample_nonzero_output_convex_subgraphs(
                graph,
                2,
                1,
                alternate_graph=alternate_graph,
                forbid_sources_and_sinks=False,
                max_states_expanded=32,
                max_samples=16,
                max_children_per_state=3,
                size_bin_width=1,
                forbidden_attr=None,
                body_forbidden_attr="forbidden",
            )
        }
        grown = {
            frozenset(nodes)
            for nodes in grow_nonzero_output_convex_subgraphs(
                graph,
                {"a"},
                alternate_graph=alternate_graph,
                max_num_inputs=2,
                max_num_outputs=1,
                forbid_sources_and_sinks=False,
                forbidden_attr=None,
                body_forbidden_attr="forbidden",
            )
        }

        self.assertIn(frozenset({"a", "b", "c"}), samples)
        self.assertNotIn(frozenset({"a", "c"}), samples)
        self.assertSetEqual(
            {frozenset({"a"}), frozenset({"a", "b", "c"})},
            grown,
        )

    def test_nonzero_output_boundary_pair_sampling_jumps_to_deep_region(self) -> None:
        graph = nx.DiGraph()
        graph.add_node("i", forbidden=True)
        graph.add_node("sink", forbidden=True)
        graph.add_edges_from(
            [
                ("i", "a"),
                ("a", "b"),
                ("a", "side"),
                ("b", "out"),
                ("side", "out"),
                ("out", "sink"),
            ]
        )

        samples = {
            frozenset(nodes)
            for nodes in sample_nonzero_output_convex_subgraphs(
                graph,
                1,
                1,
                forbid_sources_and_sinks=False,
                max_states_expanded=0,
                max_samples=8,
                boundary_pair_samples=8,
                forbidden_attr=None,
                body_forbidden_attr="forbidden",
            )
        }

        self.assertIn(frozenset({"a", "b", "side", "out"}), samples)
