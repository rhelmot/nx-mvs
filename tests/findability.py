from __future__ import annotations

from collections.abc import Hashable, Iterable
from dataclasses import dataclass
from functools import lru_cache
from os import PathLike
from pathlib import Path
from typing import Literal, TypeVar, cast

import networkx as nx

from mvs import ConvexSubgraphQuery, graph_to_input


NodeT = TypeVar("NodeT", bound=Hashable)
GraphSource = nx.DiGraph | str | PathLike[str]
Ordering = Literal["default", "sort", "toposort"]


@dataclass(frozen=True)
class FindabilityResult:
    found: bool
    minimum_growth_steps: int | None
    minimum_growth_seed: frozenset[Hashable] | None
    expected_random_growth_steps: float | None
    expected_random_seed: frozenset[Hashable] | None
    random_search_success_probability: float
    sample_count: int
    launch_point_count: int


def _load_graph(source: GraphSource | None) -> nx.DiGraph | None:
    if source is None:
        return None
    if isinstance(source, nx.DiGraph):
        return source
    return nx.DiGraph(nx.nx_pydot.read_dot(Path(source)))


def _topological_order(num_nodes: int, edges: list[tuple[int, int]]) -> list[int]:
    graph = nx.DiGraph()
    graph.add_nodes_from(range(num_nodes))
    graph.add_edges_from(edges)
    return list(nx.topological_sort(graph))


class _IndexedZeroOutputGrowthSpace:
    def __init__(self, num_nodes: int, edges: list[tuple[int, int]], alternate_edges: list[tuple[int, int]], forbidden: list[int], forbid_sources_and_sinks: bool) -> None:
        self.num_nodes = num_nodes
        self.all_mask = (1 << num_nodes) - 1
        self.pred_main = [0] * num_nodes
        self.succ_main = [0] * num_nodes
        self.pred_alt = [0] * num_nodes
        self.succ_alt = [0] * num_nodes
        self.undirected_main = [0] * num_nodes

        for source, target in edges:
            self.succ_main[source] |= 1 << target
            self.pred_main[target] |= 1 << source
            self.undirected_main[source] |= 1 << target
            self.undirected_main[target] |= 1 << source

        for source, target in alternate_edges:
            self.succ_alt[source] |= 1 << target
            self.pred_alt[target] |= 1 << source

        self.pred_tc_main, self.succ_tc_main = self._transitive_closure(
            self.pred_main,
            self.succ_main,
            _topological_order(num_nodes, edges),
        )
        if alternate_edges:
            self.pred_tc_alt, self.succ_tc_alt = self._transitive_closure(
                self.pred_alt,
                self.succ_alt,
                _topological_order(num_nodes, alternate_edges),
            )
        else:
            self.pred_tc_alt = None
            self.succ_tc_alt = None

        forbidden_mask = 0
        for index, is_forbidden in enumerate(forbidden):
            if is_forbidden:
                forbidden_mask |= 1 << index
        if forbid_sources_and_sinks:
            for node in range(num_nodes):
                if self.pred_main[node] == 0 or self.succ_main[node] == 0:
                    forbidden_mask |= 1 << node
        self.forbidden_mask = forbidden_mask

        self.singleton_closure = [0] * num_nodes
        self.singleton_augmented = [0] * num_nodes
        self.singleton_body_neighbors = [0] * num_nodes
        self.singleton_input_neighbors = [0] * num_nodes
        self.singleton_valid = [False] * num_nodes
        for node in range(num_nodes):
            closed = self.zero_output_closure(1 << node)
            self.singleton_closure[node] = closed
            augmented = self.augmented_nodes(closed)
            self.singleton_augmented[node] = augmented
            inputs = augmented & ~closed
            current = closed
            while current:
                lsb = current & -current
                index = lsb.bit_length() - 1
                self.singleton_body_neighbors[node] |= self.pred_main[index]
                self.singleton_body_neighbors[node] |= self.succ_main[index]
                current ^= lsb
            current = inputs
            while current:
                lsb = current & -current
                index = lsb.bit_length() - 1
                self.singleton_input_neighbors[node] |= self.pred_main[index]
                self.singleton_input_neighbors[node] |= self.succ_main[index]
                current ^= lsb
            self.singleton_valid[node] = (closed & self.forbidden_mask) == 0

    def _transitive_closure(
        self,
        pred_masks: list[int],
        succ_masks: list[int],
        topo_order: list[int],
    ) -> tuple[list[int], list[int]]:
        succ_tc = succ_masks[:]
        for node in reversed(topo_order):
            current = succ_masks[node]
            while current:
                lsb = current & -current
                succ = lsb.bit_length() - 1
                succ_tc[node] |= succ_tc[succ]
                current ^= lsb

        pred_tc = pred_masks[:]
        for node in topo_order:
            current = pred_masks[node]
            while current:
                lsb = current & -current
                pred = lsb.bit_length() - 1
                pred_tc[node] |= pred_tc[pred]
                current ^= lsb
        return pred_tc, succ_tc

    def augmented_nodes(self, nodes: int) -> int:
        augmented = nodes
        current = nodes
        while current:
            lsb = current & -current
            index = lsb.bit_length() - 1
            augmented |= self.pred_main[index]
            current ^= lsb
        return augmented

    def closure_in_graph(
        self,
        nodes: int,
        *,
        pred_tc: list[int],
        succ_tc: list[int],
    ) -> int:
        closed = nodes
        outside = self.all_mask & ~closed
        while outside:
            lsb = outside & -outside
            index = lsb.bit_length() - 1
            if (pred_tc[index] & closed) and (succ_tc[index] & closed):
                closed |= lsb
            outside ^= lsb
        return closed

    def dual_closure(self, nodes: int) -> int:
        closed = nodes
        while True:
            next_nodes = self.closure_in_graph(
                closed,
                pred_tc=self.pred_tc_main,
                succ_tc=self.succ_tc_main,
            )
            if self.pred_tc_alt is not None and self.succ_tc_alt is not None:
                next_nodes |= self.closure_in_graph(
                    closed,
                    pred_tc=self.pred_tc_alt,
                    succ_tc=self.succ_tc_alt,
                )
            if next_nodes == closed:
                return next_nodes
            closed = next_nodes

    def zero_output_closure(self, nodes: int) -> int:
        closed = nodes
        while True:
            next_nodes = closed
            current = closed
            while current:
                lsb = current & -current
                index = lsb.bit_length() - 1
                next_nodes |= self.succ_tc_main[index]
                current ^= lsb
            if self.pred_tc_alt is not None and self.succ_tc_alt is not None:
                next_nodes = self.dual_closure(next_nodes)
            if next_nodes == closed:
                return next_nodes
            closed = next_nodes

    def input_mask(self, nodes: int) -> int:
        inputs = 0
        current = nodes
        while current:
            lsb = current & -current
            index = lsb.bit_length() - 1
            inputs |= self.pred_main[index]
            current ^= lsb
        return inputs & ~nodes

    def num_outputs(self, nodes: int) -> int:
        outputs = 0
        current = nodes
        while current:
            lsb = current & -current
            index = lsb.bit_length() - 1
            if self.succ_main[index] & ~nodes:
                outputs += 1
            current ^= lsb
        return outputs

    def is_connected_with_inputs(self, nodes: int) -> bool:
        inputs = self.input_mask(nodes)
        augmented = nodes | inputs
        if augmented == 0:
            return True
        start_bit = augmented & -augmented
        start = start_bit.bit_length() - 1
        seen = 1 << start
        stack = [start]
        while stack:
            node = stack.pop()
            neighbors = self.undirected_main[node] & augmented & ~seen
            while neighbors:
                lsb = neighbors & -neighbors
                neighbor = lsb.bit_length() - 1
                if (inputs & (1 << node)) and (inputs & (1 << neighbor)):
                    neighbors ^= lsb
                    continue
                seen |= lsb
                stack.append(neighbor)
                neighbors ^= lsb
        return seen == augmented

    def can_connect(self, singleton_node: int, current_nodes: int, current_augmented: int) -> bool:
        return (
            (self.singleton_augmented[singleton_node] & current_augmented) != 0
            or (self.singleton_body_neighbors[singleton_node] & current_augmented) != 0
            or (self.singleton_input_neighbors[singleton_node] & current_nodes) != 0
        )

    def is_valid_state(self, nodes: int, *, max_num_inputs: int, max_subgraph_size: int | None) -> bool:
        if nodes & self.forbidden_mask:
            return False
        if max_subgraph_size is not None and nodes.bit_count() > max_subgraph_size:
            return False
        if self.input_mask(nodes).bit_count() > max_num_inputs:
            return False
        if self.num_outputs(nodes) != 0:
            return False
        return self.is_connected_with_inputs(nodes)

    def is_canonical_target(self, nodes: int, *, max_num_inputs: int, max_subgraph_size: int | None) -> bool:
        return self.zero_output_closure(nodes) == nodes and self.is_valid_state(
            nodes,
            max_num_inputs=max_num_inputs,
            max_subgraph_size=max_subgraph_size,
        )

    def children(self, nodes: int, target_mask: int, *, max_num_inputs: int, max_subgraph_size: int | None) -> tuple[int, ...]:
        current_augmented = self.augmented_nodes(nodes)
        seen: set[int] = set()
        next_states: list[int] = []
        remaining = target_mask & ~nodes
        while remaining:
            lsb = remaining & -remaining
            node = lsb.bit_length() - 1
            remaining ^= lsb

            if not self.singleton_valid[node]:
                continue
            if self.singleton_closure[node] & ~nodes == 0:
                continue
            if not self.can_connect(node, nodes, current_augmented):
                continue

            next_state = self.zero_output_closure(nodes | (1 << node))
            if next_state == nodes:
                continue
            if next_state & ~target_mask:
                continue
            if not self.is_valid_state(
                next_state,
                max_num_inputs=max_num_inputs,
                max_subgraph_size=max_subgraph_size,
            ):
                continue
            if next_state in seen:
                continue
            seen.add(next_state)
            next_states.append(next_state)
        return tuple(sorted(next_states, key=lambda state: (state.bit_count(), state)))


def measure_findability(
    graph: GraphSource = "graph.dot",
    alternate_graph: GraphSource | None = None,
    *,
    target_nodes: Iterable[Hashable],
    max_num_inputs: int = 4,
    max_subgraph_size: int = 50,
    forbid_sources_and_sinks: bool = False,
    forbidden_attr: str | None = "forbidden",
    body_forbidden_attr: str | None = None,
    input_forbidden_attr: str | None = None,
    ordering: Ordering = "toposort",
    max_states_expanded: int = 10_000,
    max_samples: int = 1_000,
    max_children_per_state: int = 2,
    size_bin_width: int = 4,
    thicken_radius: int = 1,
    bucket_by_num_inputs: bool = True,
    minimal_node_bin_width: int = 1,
    sampling_passes: int = 1,
    exact_kernel_size: int = 0,
) -> FindabilityResult:
    """
    Measure whether a zero-output dual-convex target is reachable from the sampled
    launch-point index, and how hard it is to reach under native growth semantics.

    The result reports:
    - the minimum number of one-step growth transitions from any sampled seed
    - a conditional expected number of transitions under uniform random child
      selection from the best random seed
    - the success probability for that random policy

    This is scoped to the current zero-output connected-through-inputs search
    regime used by the sampler and grower.
    """

    loaded_graph = cast("nx.DiGraph[Hashable]", _load_graph(graph))
    loaded_alternate_graph = cast(
        "nx.DiGraph[Hashable] | None",
        _load_graph(alternate_graph),
    )
    payload, node_order = graph_to_input(
        loaded_graph,
        alternate_graph=loaded_alternate_graph,
        forbidden_attr=forbidden_attr,
        body_forbidden_attr=body_forbidden_attr,
        input_forbidden_attr=input_forbidden_attr,
        forbid_sources_and_sinks=forbid_sources_and_sinks,
        ordering=ordering,
    )
    node_index = {node: index for index, node in enumerate(node_order)}
    target_mask = 0
    target_set = frozenset(target_nodes)
    for node in target_set:
        if node not in node_index:
            raise ValueError(f"target node {node!r} is not present in the graph set")
        target_mask |= 1 << node_index[node]

    max_size_limit = None if max_subgraph_size < 0 else max_subgraph_size
    space = _IndexedZeroOutputGrowthSpace(
        payload.num_nodes,
        payload.edges,
        payload.alternate_edges,
        payload.forbidden,
        payload.forbid_sources_and_sinks,
    )
    if not space.is_canonical_target(
        target_mask,
        max_num_inputs=max_num_inputs,
        max_subgraph_size=max_size_limit,
    ):
        raise ValueError(
            "target_nodes must describe a canonical valid zero-output dual-convex "
            "subgraph under the supplied constraints"
        )

    samples = [
        frozenset(sample)
        for sample in ConvexSubgraphQuery(
            max_subgraph_size=max_subgraph_size,
            forbid_sources_and_sinks=forbid_sources_and_sinks,
            forbidden_attr=forbidden_attr,
            body_forbidden_attr=body_forbidden_attr,
            input_forbidden_attr=input_forbidden_attr,
            ordering=ordering,
            sampling_max_states_expanded=max_states_expanded,
            sampling_max_samples=max_samples,
            sampling_max_children_per_state=max_children_per_state,
            sampling_size_bin_width=size_bin_width,
            sampling_thicken_radius=thicken_radius,
            sampling_bucket_by_num_inputs=bucket_by_num_inputs,
            sampling_minimal_node_bin_width=minimal_node_bin_width,
            sampling_passes=sampling_passes,
            sampling_exact_kernel_size=exact_kernel_size,
        ).sample(
            loaded_graph,
            max_num_inputs=max_num_inputs,
            max_num_outputs=0,
            alternate_graph=loaded_alternate_graph,
        )
    ]

    sample_masks = [
        sum(1 << node_index[node] for node in sample)
        for sample in samples
    ]
    launch_masks = [
        mask
        for mask in sample_masks
        if mask & ~target_mask == 0
    ]

    @lru_cache(maxsize=None)
    def metrics(nodes: int) -> tuple[bool, int | None, float, float | None]:
        if nodes == target_mask:
            return True, 0, 1.0, 0.0

        children = space.children(
            nodes,
            target_mask,
            max_num_inputs=max_num_inputs,
            max_subgraph_size=max_size_limit,
        )
        if not children:
            return False, None, 0.0, None

        child_metrics = [metrics(child) for child in children]
        reachable_children = [child for child, child_metric in zip(children, child_metrics) if child_metric[0]]
        if not reachable_children:
            return False, None, 0.0, None

        minimum_growth_steps = 1 + min(
            cast(int, child_metric[1])
            for child_metric in child_metrics
            if child_metric[0]
        )
        success_probability = sum(child_metric[2] for child_metric in child_metrics) / len(children)

        success_mass = sum(child_metric[2] for child_metric in child_metrics if child_metric[2] > 0.0)
        conditional_expected_steps = sum(
            (child_metric[2] / success_mass) * (1.0 + cast(float, child_metric[3]))
            for child_metric in child_metrics
            if child_metric[2] > 0.0
        )
        return True, minimum_growth_steps, success_probability, conditional_expected_steps

    reachable_launch_masks = [
        mask for mask in launch_masks if metrics(mask)[0]
    ]
    if not reachable_launch_masks:
        return FindabilityResult(
            found=False,
            minimum_growth_steps=None,
            minimum_growth_seed=None,
            expected_random_growth_steps=None,
            expected_random_seed=None,
            random_search_success_probability=0.0,
            sample_count=len(samples),
            launch_point_count=len(launch_masks),
        )

    best_exact_mask = min(
        reachable_launch_masks,
        key=lambda mask: (
            cast(int, metrics(mask)[1]),
            cast(float, metrics(mask)[3]),
            -mask.bit_count(),
            mask,
        ),
    )
    best_random_mask = min(
        reachable_launch_masks,
        key=lambda mask: (
            cast(float, metrics(mask)[3]),
            cast(int, metrics(mask)[1]),
            -mask.bit_count(),
            mask,
        ),
    )

    def to_nodes(mask: int) -> frozenset[Hashable]:
        return frozenset(
            node_order[index]
            for index in range(payload.num_nodes)
            if mask & (1 << index)
        )

    return FindabilityResult(
        found=True,
        minimum_growth_steps=cast(int, metrics(best_exact_mask)[1]),
        minimum_growth_seed=to_nodes(best_exact_mask),
        expected_random_growth_steps=cast(float, metrics(best_random_mask)[3]),
        expected_random_seed=to_nodes(best_random_mask),
        random_search_success_probability=metrics(best_random_mask)[2],
        sample_count=len(samples),
        launch_point_count=len(launch_masks),
    )
