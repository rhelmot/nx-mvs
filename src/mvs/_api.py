from __future__ import annotations

from collections.abc import Callable, Hashable, Iterator
import math
from typing import Literal, TypeVar, cast, overload

import networkx as nx

from ._native import (
    GraphInput,
    grow_zero_output_graph_input,
    iter_all_graph_input,
    sample_zero_output_graph_input,
    solve_graph_input,
)


NodeT = TypeVar("NodeT", bound=Hashable)
StateT = TypeVar("StateT")
Ordering = Literal["default", "sort", "toposort"]


def _as_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "on"}:
            return True
        if normalized in {"0", "false", "no", "off", ""}:
            return False
    return bool(value)


def _subgraph_weight(
    graph: nx.DiGraph[NodeT],
    subgraph: set[NodeT],
    *,
    weighted: bool,
    weight_attr: str,
) -> float:
    if not weighted:
        return float(len(subgraph))
    return sum(float(graph.nodes[node].get(weight_attr, 1.0)) for node in subgraph)


def graph_to_input(
    graph: nx.DiGraph[NodeT],
    *,
    alternate_graph: nx.DiGraph[NodeT] | None = None,
    weighted: bool = False,
    weight_attr: str = "weight",
    forbidden_attr: str = "forbidden",
    forbid_sources_and_sinks: bool = True,
    ordering: Ordering = "toposort",
    name: str | None = None,
) -> tuple[GraphInput, tuple[NodeT, ...]]:  # second tuple item is the reverse mapping for the ints
    if not nx.is_directed_acyclic_graph(graph):
        raise ValueError("Graph must be a DAG")
    graph_nodes = set(graph.nodes)
    alternate_nodes: set[NodeT] = set()
    graph_forbidden = {
        node for node in graph_nodes if _as_bool(graph.nodes[node].get(forbidden_attr, False))
    }
    alternate_forbidden: set[NodeT] = set()
    if alternate_graph is not None:
        if not nx.is_directed_acyclic_graph(alternate_graph):
            raise ValueError("alternate_graph must be a DAG")
        alternate_nodes = set(alternate_graph.nodes)
        alternate_forbidden = {
            node
            for node in alternate_nodes
            if _as_bool(alternate_graph.nodes[node].get(forbidden_attr, False))
        }
        missing_from_alternate = graph_nodes - alternate_nodes
        if missing_from_alternate - graph_forbidden:
            raise ValueError(
                "alternate_graph may only omit nodes that are forbidden in graph"
            )
        missing_from_graph = alternate_nodes - graph_nodes
        if missing_from_graph - alternate_forbidden:
            raise ValueError(
                "graph may only omit nodes that are forbidden in alternate_graph"
            )
    all_nodes = graph_nodes | alternate_nodes

    match ordering:
        case "default":
            node_order = list(graph)
            if alternate_graph is not None:
                node_order.extend(
                    node for node in alternate_graph if node not in graph_nodes
                )
        case "sort":
            # if it crashes it crashes
            node_order = sorted(all_nodes)  # type: ignore
        case "toposort":
            node_order = list(nx.topological_sort(graph))
            if alternate_graph is not None:
                node_order.extend(
                    node
                    for node in nx.topological_sort(alternate_graph)
                    if node not in graph_nodes
                )
    node_index = {node: index for index, node in enumerate(node_order)}
    forbidden_nodes = graph_forbidden | alternate_forbidden

    def node_weight(node: NodeT) -> float:
        if not weighted:
            return 1.0
        if node in graph_nodes:
            return float(graph.nodes[cast("NodeT", node)].get(weight_attr, 1.0))
        return float(
            cast("nx.DiGraph[NodeT]", alternate_graph).nodes[cast("NodeT", node)].get(
                weight_attr, 1.0
            )
        )

    payload = GraphInput()
    payload.name = graph.graph.get("name", "") if name is None else name
    payload.frequency = int(graph.graph.get("frequency", 0))
    payload.num_nodes = len(node_order)
    payload.edges = [
        (node_index[source], node_index[target]) for source, target in graph.edges()
    ]
    payload.alternate_edges = (
        [
            (node_index[source], node_index[target])
            for source, target in alternate_graph.edges()
        ]
        if alternate_graph is not None
        else []
    )
    payload.weights = [node_weight(node) for node in node_order]
    payload.forbid_sources_and_sinks = forbid_sources_and_sinks
    payload.forbidden = [
        1 if node in forbidden_nodes else 0
        for node in node_order
    ]
    return payload, tuple(node_order)


def enumerate_maximum_convex_subgraphs(
    graph: nx.DiGraph[NodeT],
    max_num_inputs: int,
    max_num_outputs: int,
    *,
    alternate_graph: nx.DiGraph[NodeT] | None = None,
    max_subgraph_size: int | None = None,
    weighted: bool = False,
    weight_attr: str = "weight",
    forbidden_attr: str = "forbidden",
    forbid_sources_and_sinks: bool = True,
    allow_zero_outputs: bool = False,
    connected_only: bool = False,
    iteration_type: str = "linear-rev",
    ordering: Ordering = "toposort",
    flags: int = 0xFF,
) -> Iterator[set[NodeT]]:
    """
    Enumerate the maximum directed convex subgraphs of a DAG under I/O constraints.

    This algorithm is the implementation for Giaquinta et. al. "Maximum Convex Subgraphs Under I/O Constraint for
    Automatic Identification of Custom Instructions"
    """

    native_max_subgraph_size = -1 if max_subgraph_size is None else max_subgraph_size
    if native_max_subgraph_size < -1:
        raise ValueError("max_subgraph_size must be non-negative or None")

    if max_num_outputs <= 1:
        best_weight = float("-inf")
        best_subgraphs: list[set[NodeT]] = []
        for subgraph in enumerate_convex_subgraphs(
            graph,
            max_num_inputs,
            max_num_outputs,
            alternate_graph=alternate_graph,
            max_subgraph_size=max_subgraph_size,
            weighted=weighted,
            weight_attr=weight_attr,
            forbidden_attr=forbidden_attr,
            forbid_sources_and_sinks=forbid_sources_and_sinks,
            allow_zero_outputs=allow_zero_outputs,
            connected_only=connected_only,
            ordering=ordering,
        ):
            weight = _subgraph_weight(
                graph,
                subgraph,
                weighted=weighted,
                weight_attr=weight_attr,
            )
            if weight > best_weight:
                best_weight = weight
                best_subgraphs = [subgraph]
            elif math.isclose(weight, best_weight):
                best_subgraphs.append(subgraph)
        return iter(best_subgraphs)

    if alternate_graph is not None:
        raise NotImplementedError(
            "alternate_graph is currently only supported for exhaustive enumeration "
            "and maximum enumeration when max_num_outputs <= 1"
        )

    if connected_only:
        best_weight = float("-inf")
        best_subgraphs: list[set[NodeT]] = []
        for component_nodes in _connected_component_node_sets(graph, alternate_graph):
            component = cast("nx.DiGraph[NodeT]", graph.subgraph(component_nodes).copy())
            payload, node_order = graph_to_input(
                component,
                alternate_graph=(
                    cast("nx.DiGraph[NodeT]", alternate_graph.subgraph(component_nodes).copy())
                    if alternate_graph is not None
                    else None
                ),
                weighted=weighted,
                weight_attr=weight_attr,
                forbidden_attr=forbidden_attr,
                forbid_sources_and_sinks=forbid_sources_and_sinks,
                ordering=ordering,
            )
            result = solve_graph_input(
                payload,
                max_num_inputs,
                max_num_outputs,
                native_max_subgraph_size,
                iteration_type=iteration_type,
                flags=flags,
            )
            if not result.subgraphs:
                continue
            if result.max_weight > best_weight:
                best_weight = result.max_weight
                best_subgraphs = [
                    {node_order[index] for index in subgraph}
                    for subgraph in result.subgraphs
                ]
            elif math.isclose(result.max_weight, best_weight):
                best_subgraphs.extend(
                    {node_order[index] for index in subgraph}
                    for subgraph in result.subgraphs
                )
        return iter(best_subgraphs)

    payload, node_order = graph_to_input(
        graph,
        alternate_graph=alternate_graph,
        weighted=weighted,
        weight_attr=weight_attr,
        forbidden_attr=forbidden_attr,
        forbid_sources_and_sinks=forbid_sources_and_sinks,
        ordering=ordering,
    )
    result = solve_graph_input(
        payload,
        max_num_inputs,
        max_num_outputs,
        native_max_subgraph_size,
        iteration_type=iteration_type,
        flags=flags,
    )
    return ({node_order[index] for index in subgraph} for subgraph in result.subgraphs)


def enumerate_convex_subgraphs(
    graph: nx.DiGraph[NodeT],
    max_num_inputs: int,
    max_num_outputs: int,
    *,
    alternate_graph: nx.DiGraph[NodeT] | None = None,
    max_subgraph_size: int | None = None,
    weighted: bool = False,
    weight_attr: str = "weight",
    forbidden_attr: str = "forbidden",
    forbid_sources_and_sinks: bool = True,
    allow_zero_outputs: bool = False,
    connected_only: bool = False,
    ordering: Ordering = "toposort",
    max_queue_size: int = 128,
) -> Iterator[set[NodeT]]:
    """
    Enumerate all directed convex subgraphs of a DAG under I/O constraints.

    This uses the upstream exhaustive enumerator rather than the maximum-only search.
    """

    native_max_subgraph_size = -1 if max_subgraph_size is None else max_subgraph_size
    if native_max_subgraph_size < -1:
        raise ValueError("max_subgraph_size must be non-negative or None")

    if connected_only:
        def iter_connected_results() -> Iterator[set[NodeT]]:
            for component_nodes in _connected_component_node_sets(graph, alternate_graph):
                component = cast("nx.DiGraph[NodeT]", graph.subgraph(component_nodes).copy())
                payload, node_order = graph_to_input(
                    component,
                    alternate_graph=(
                        cast("nx.DiGraph[NodeT]", alternate_graph.subgraph(component_nodes).copy())
                        if alternate_graph is not None
                        else None
                    ),
                    weighted=weighted,
                    weight_attr=weight_attr,
                    forbidden_attr=forbidden_attr,
                    forbid_sources_and_sinks=forbid_sources_and_sinks,
                    ordering=ordering,
                )
                yield from _iter_convex_subgraphs(
                    component,
                    payload,
                    node_order,
                    max_num_inputs=max_num_inputs,
                    max_num_outputs=max_num_outputs,
                    max_subgraph_size=native_max_subgraph_size,
                    weighted=weighted,
                    weight_attr=weight_attr,
                    forbidden_attr=forbidden_attr,
                    forbid_sources_and_sinks=forbid_sources_and_sinks,
                    allow_zero_outputs=allow_zero_outputs,
                    connected_only=connected_only,
                    ordering=ordering,
                    max_queue_size=max_queue_size,
                )

        return iter_connected_results()

    payload, node_order = graph_to_input(
        graph,
        alternate_graph=alternate_graph,
        weighted=weighted,
        weight_attr=weight_attr,
        forbidden_attr=forbidden_attr,
        forbid_sources_and_sinks=forbid_sources_and_sinks,
        ordering=ordering,
    )

    return _iter_convex_subgraphs(
        graph,
        payload,
        node_order,
        max_num_inputs=max_num_inputs,
        max_num_outputs=max_num_outputs,
        max_subgraph_size=native_max_subgraph_size,
        weighted=weighted,
        weight_attr=weight_attr,
        forbidden_attr=forbidden_attr,
        forbid_sources_and_sinks=forbid_sources_and_sinks,
        allow_zero_outputs=allow_zero_outputs,
        connected_only=connected_only,
        ordering=ordering,
        max_queue_size=max_queue_size,
    )


def sample_zero_output_convex_subgraphs(
    graph: nx.DiGraph[NodeT],
    max_num_inputs: int,
    *,
    alternate_graph: nx.DiGraph[NodeT] | None = None,
    max_subgraph_size: int | None = None,
    weighted: bool = False,
    weight_attr: str = "weight",
    forbidden_attr: str = "forbidden",
    forbid_sources_and_sinks: bool = True,
    ordering: Ordering = "toposort",
    max_states_expanded: int = 10000,
    max_samples: int = 1000,
    max_children_per_state: int = 2,
    size_bin_width: int = 4,
    thicken_radius: int = 1,
    bucket_by_num_inputs: bool = True,
    minimal_node_bin_width: int = 1,
    sampling_passes: int = 1,
    exact_kernel_size: int = 0,
) -> Iterator[set[NodeT]]:
    """
    Heuristically sample connected zero-output convex subgraphs.

    This is not exhaustive. It targets the expensive zero-output connected path by
    exploring a bounded subset of the native state DAG and favoring underrepresented
    size regions at branch points.
    """

    native_max_subgraph_size = -1 if max_subgraph_size is None else max_subgraph_size
    if native_max_subgraph_size < -1:
        raise ValueError("max_subgraph_size must be non-negative or None")
    if max_states_expanded < 0 or max_samples < 0:
        raise ValueError("sampling budgets must be non-negative")
    if max_children_per_state <= 0:
        raise ValueError("max_children_per_state must be positive")
    if size_bin_width <= 0:
        raise ValueError("size_bin_width must be positive")
    if thicken_radius < 0:
        raise ValueError("thicken_radius must be non-negative")
    if minimal_node_bin_width < 0:
        raise ValueError("minimal_node_bin_width must be non-negative")
    if sampling_passes <= 0:
        raise ValueError("sampling_passes must be positive")
    if exact_kernel_size < 0:
        raise ValueError("exact_kernel_size must be non-negative")

    pass_configs = _sampling_pass_configs(
        ordering,
        max_children_per_state=max_children_per_state,
        size_bin_width=size_bin_width,
        pass_count=sampling_passes,
    )
    per_pass_states = max(1, math.ceil(max_states_expanded / len(pass_configs)))
    per_pass_samples = max(1, math.ceil(max_samples / len(pass_configs)))

    def iter_component_samples() -> Iterator[set[NodeT]]:
        for component_nodes in _connected_component_node_sets(graph, alternate_graph):
            component = cast("nx.DiGraph[NodeT]", graph.subgraph(component_nodes).copy())
            component_alternate = (
                cast("nx.DiGraph[NodeT]", alternate_graph.subgraph(component_nodes).copy())
                if alternate_graph is not None
                else None
            )
            seen: set[frozenset[NodeT]] = set()

            if exact_kernel_size > 0:
                exact_limit = (
                    exact_kernel_size
                    if native_max_subgraph_size < 0
                    else min(native_max_subgraph_size, exact_kernel_size)
                )
                for subgraph in enumerate_convex_subgraphs(
                    component,
                    max_num_inputs,
                    0,
                    alternate_graph=component_alternate,
                    max_subgraph_size=exact_limit,
                    weighted=weighted,
                    weight_attr=weight_attr,
                    forbidden_attr=forbidden_attr,
                    forbid_sources_and_sinks=forbid_sources_and_sinks,
                    allow_zero_outputs=False,
                    connected_only=True,
                    ordering=ordering,
                ):
                    key = frozenset(subgraph)
                    if key in seen:
                        continue
                    seen.add(key)
                    yield subgraph

            for pass_ordering, pass_children, pass_size_bin_width in pass_configs:
                payload, node_order = graph_to_input(
                    component,
                    alternate_graph=component_alternate,
                    weighted=weighted,
                    weight_attr=weight_attr,
                    forbidden_attr=forbidden_attr,
                    forbid_sources_and_sinks=forbid_sources_and_sinks,
                    ordering=pass_ordering,
                )
                result = sample_zero_output_graph_input(
                    payload,
                    max_num_inputs,
                    native_max_subgraph_size,
                    max_states_expanded=per_pass_states,
                    max_samples=per_pass_samples,
                    max_children_per_state=pass_children,
                    size_bin_width=pass_size_bin_width,
                    thicken_radius=thicken_radius,
                    bucket_by_num_inputs=bucket_by_num_inputs,
                    minimal_node_bin_width=minimal_node_bin_width,
                )
                for subgraph in result.subgraphs:
                    nodes = {node_order[index] for index in subgraph}
                    if exact_kernel_size > 0 and len(nodes) <= exact_kernel_size:
                        continue
                    key = frozenset(nodes)
                    if key in seen:
                        continue
                    seen.add(key)
                    yield nodes

    return iter_component_samples()


@overload
def grow_zero_output_convex_subgraphs(
    graph: nx.DiGraph[NodeT],
    seed_nodes: set[NodeT],
    *,
    alternate_graph: nx.DiGraph[NodeT] | None = None,
    max_num_inputs: int = 4,
    max_subgraph_size: int | None = None,
    forbidden_attr: str = "forbidden",
    forbid_sources_and_sinks: bool = False,
    ordering: Ordering = "toposort",
    oracle: None = None,
    initial_oracle_state: None = None,
) -> Iterator[set[NodeT]]: ...


@overload
def grow_zero_output_convex_subgraphs(
    graph: nx.DiGraph[NodeT],
    seed_nodes: set[NodeT],
    *,
    alternate_graph: nx.DiGraph[NodeT] | None = None,
    max_num_inputs: int = 4,
    max_subgraph_size: int | None = None,
    forbidden_attr: str = "forbidden",
    forbid_sources_and_sinks: bool = False,
    ordering: Ordering = "toposort",
    oracle: Callable[[StateT, set[NodeT]], StateT | None],
    initial_oracle_state: StateT,
) -> Iterator[set[NodeT]]: ...


def grow_zero_output_convex_subgraphs(
    graph: nx.DiGraph[NodeT],
    seed_nodes: set[NodeT],
    *,
    alternate_graph: nx.DiGraph[NodeT] | None = None,
    max_num_inputs: int = 4,
    max_subgraph_size: int | None = None,
    forbidden_attr: str = "forbidden",
    forbid_sources_and_sinks: bool = False,
    ordering: Ordering = "toposort",
    oracle: Callable[..., object | None] | None = None,
    initial_oracle_state: object | None = None,
) -> Iterator[set[NodeT]]:
    native_max_subgraph_size = -1 if max_subgraph_size is None else max_subgraph_size
    if native_max_subgraph_size < -1:
        raise ValueError("max_subgraph_size must be non-negative or None")

    payload, node_order = graph_to_input(
        graph,
        alternate_graph=alternate_graph,
        forbidden_attr=forbidden_attr,
        forbid_sources_and_sinks=forbid_sources_and_sinks,
        ordering=ordering,
    )
    node_index = {node: index for index, node in enumerate(node_order)}
    seed_indices: list[int] = []
    for node in seed_nodes:
        try:
            seed_indices.append(node_index[node])
        except KeyError as exc:
            raise ValueError(f"seed node {node!r} is not present in the graph set") from exc

    def oracle_indices(state: object | None, indices: list[int]) -> object | None:
        nodes = {node_order[index] for index in indices}
        assert oracle is not None
        return oracle(state, nodes)

    result = grow_zero_output_graph_input(
        payload,
        seed_indices,
        max_num_inputs,
        native_max_subgraph_size,
        oracle_indices if oracle is not None else None,
        initial_oracle_state,
    )
    return ({node_order[index] for index in subgraph} for subgraph in result.subgraphs)


def _iter_convex_subgraphs(
    graph: nx.DiGraph[NodeT],
    payload: GraphInput,
    node_order: tuple[NodeT, ...],
    *,
    max_num_inputs: int,
    max_num_outputs: int,
    max_subgraph_size: int,
    weighted: bool,
    weight_attr: str,
    forbidden_attr: str,
    forbid_sources_and_sinks: bool,
    allow_zero_outputs: bool,
    connected_only: bool,
    ordering: Ordering,
    max_queue_size: int,
) -> Iterator[set[NodeT]]:
    seen: set[tuple[int, ...]] | None = set() if allow_zero_outputs else None

    for subgraph in iter_all_graph_input(
        payload,
        max_num_inputs,
        max_num_outputs,
        max_subgraph_size,
        max_queue_size=max_queue_size,
        connected_only=connected_only,
    ):
        if seen is not None:
            key = tuple(subgraph)
            seen.add(key)
        yield {node_order[index] for index in subgraph}

    if allow_zero_outputs:
        for subgraph in iter_all_graph_input(
            payload,
            max_num_inputs,
            0,
            max_subgraph_size,
            max_queue_size=max_queue_size,
            connected_only=connected_only,
        ):
            key = tuple(subgraph)
            if seen is not None and key in seen:
                continue
            yield {node_order[index] for index in subgraph}


def _connected_component_node_sets(
    graph: nx.DiGraph[NodeT],
    alternate_graph: nx.DiGraph[NodeT] | None = None,
) -> tuple[set[NodeT], ...]:
    if graph.number_of_nodes() == 0:
        return ()
    if alternate_graph is None:
        if nx.is_weakly_connected(graph):
            return (set(graph.nodes),)
        return tuple(set(nodes) for nodes in nx.weakly_connected_components(graph))

    combined = nx.Graph()
    combined.add_nodes_from(graph.nodes)
    combined.add_edges_from(graph.edges())
    combined.add_nodes_from(alternate_graph.nodes)
    combined.add_edges_from(alternate_graph.edges())
    if combined.number_of_nodes() == 0:
        return ()
    if nx.is_connected(combined):
        return (set(combined.nodes),)
    return tuple(set(nodes) for nodes in nx.connected_components(combined))


def _sampling_pass_configs(
    base_ordering: Ordering,
    *,
    max_children_per_state: int,
    size_bin_width: int,
    pass_count: int,
) -> tuple[tuple[Ordering, int, int], ...]:
    candidates: list[tuple[Ordering, int, int]] = [
        (base_ordering, max_children_per_state, size_bin_width),
        ("sort", max_children_per_state, max(1, size_bin_width // 2)),
        ("default", max_children_per_state + 1, size_bin_width),
        ("toposort", max_children_per_state, size_bin_width + 1),
        ("sort", max_children_per_state + 1, size_bin_width),
        ("default", max_children_per_state, max(1, size_bin_width // 2)),
    ]

    unique_configs = tuple(
        dict.fromkeys(candidates)
    )
    return unique_configs[:pass_count]
