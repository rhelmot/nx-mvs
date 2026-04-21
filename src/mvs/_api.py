from __future__ import annotations

from collections.abc import Hashable, Iterator
import math
from typing import Literal, TypeVar, cast

import networkx as nx

from ._native import GraphInput, iter_all_graph_input, solve_graph_input


NodeT = TypeVar("NodeT", bound=Hashable)


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
    ordering: Literal["default", "sort", "toposort"] = "toposort",
    name: str | None = None,
) -> tuple[GraphInput, tuple[NodeT, ...]]:  # second tuple item is the reverse mapping for the ints
    if not nx.is_directed_acyclic_graph(graph):
        raise ValueError("Graph must be a DAG")
    if alternate_graph is not None:
        if not nx.is_directed_acyclic_graph(alternate_graph):
            raise ValueError("alternate_graph must be a DAG")
        if set(alternate_graph.nodes) != set(graph.nodes):
            raise ValueError("alternate_graph must have exactly the same nodes as graph")

    match ordering:
        case "default":
            node_order = list(graph)
        case "sort":
            # if it crashes it crashes
            node_order = sorted(graph)  # type: ignore
        case "toposort":
            node_order = list(nx.topological_sort(graph))
    node_index = {node: index for index, node in enumerate(node_order)}

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
    payload.weights = [
        float(graph.nodes[node].get(weight_attr, 1.0)) if weighted else 1.0
        for node in node_order
    ]
    payload.forbid_sources_and_sinks = forbid_sources_and_sinks
    payload.forbidden = [
        1 if bool(graph.nodes[node].get(forbidden_attr, False)) else 0
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
    ordering: Literal["default", "sort", "toposort"] = "toposort",
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
        for component in _weakly_connected_component_graphs(graph):
            payload, node_order = graph_to_input(
                component,
                alternate_graph=(
                    cast("nx.DiGraph[NodeT]", alternate_graph.subgraph(component.nodes).copy())
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
    ordering: Literal["default", "sort", "toposort"] = "toposort",
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
            for component in _weakly_connected_component_graphs(graph):
                payload, node_order = graph_to_input(
                    component,
                    alternate_graph=(
                        cast("nx.DiGraph[NodeT]", alternate_graph.subgraph(component.nodes).copy())
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
    ordering: Literal["default", "sort", "toposort"],
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


def _weakly_connected_component_graphs(
    graph: nx.DiGraph[NodeT],
) -> tuple[nx.DiGraph[NodeT], ...]:
    if graph.number_of_nodes() == 0:
        return ()
    if nx.is_weakly_connected(graph):
        return (graph,)
    return tuple(
        cast("nx.DiGraph[NodeT]", graph.subgraph(nodes).copy())
        for nodes in nx.weakly_connected_components(graph)
    )
