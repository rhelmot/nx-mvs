from __future__ import annotations

from collections.abc import Hashable, Iterator
import math
from typing import Literal, TypeVar

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
    weighted: bool = False,
    weight_attr: str = "weight",
    forbidden_attr: str = "forbidden",
    forbid_sources_and_sinks: bool = True,
    ordering: Literal["default", "sort", "toposort"] = "toposort",
    name: str | None = None,
) -> tuple[GraphInput, tuple[NodeT, ...]]:  # second tuple item is the reverse mapping for the ints
    if not nx.is_directed_acyclic_graph(graph):
        raise ValueError("Graph must be a DAG")

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
    weighted: bool = False,
    weight_attr: str = "weight",
    forbidden_attr: str = "forbidden",
    forbid_sources_and_sinks: bool = True,
    allow_zero_outputs: bool = False,
    iteration_type: str = "linear-rev",
    ordering: Literal["default", "sort", "toposort"] = "toposort",
    flags: int = 0xFF,
) -> Iterator[set[NodeT]]:
    """
    Enumerate the maximum directed convex subgraphs of a DAG under I/O constraints.

    This algorithm is the implementation for Giaquinta et. al. "Maximum Convex Subgraphs Under I/O Constraint for
    Automatic Identification of Custom Instructions"
    """

    if max_num_outputs <= 1:
        best_weight = float("-inf")
        best_subgraphs: list[set[NodeT]] = []
        for subgraph in enumerate_convex_subgraphs(
            graph,
            max_num_inputs,
            max_num_outputs,
            weighted=weighted,
            weight_attr=weight_attr,
            forbidden_attr=forbidden_attr,
            forbid_sources_and_sinks=forbid_sources_and_sinks,
            allow_zero_outputs=allow_zero_outputs,
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

    payload, node_order = graph_to_input(
        graph,
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
        iteration_type=iteration_type,
        flags=flags,
    )
    return ({node_order[index] for index in subgraph} for subgraph in result.subgraphs)


def enumerate_convex_subgraphs(
    graph: nx.DiGraph[NodeT],
    max_num_inputs: int,
    max_num_outputs: int,
    *,
    weighted: bool = False,
    weight_attr: str = "weight",
    forbidden_attr: str = "forbidden",
    forbid_sources_and_sinks: bool = True,
    allow_zero_outputs: bool = False,
    ordering: Literal["default", "sort", "toposort"] = "toposort",
    max_queue_size: int = 128,
) -> Iterator[set[NodeT]]:
    """
    Enumerate all directed convex subgraphs of a DAG under I/O constraints.

    This uses the upstream exhaustive enumerator rather than the maximum-only search.
    """

    payload, node_order = graph_to_input(
        graph,
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
        weighted=weighted,
        weight_attr=weight_attr,
        forbidden_attr=forbidden_attr,
        forbid_sources_and_sinks=forbid_sources_and_sinks,
        allow_zero_outputs=allow_zero_outputs,
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
    weighted: bool,
    weight_attr: str,
    forbidden_attr: str,
    forbid_sources_and_sinks: bool,
    allow_zero_outputs: bool,
    ordering: Literal["default", "sort", "toposort"],
    max_queue_size: int,
) -> Iterator[set[NodeT]]:
    for subgraph in iter_all_graph_input(
        payload,
        max_num_inputs,
        max_num_outputs,
        max_queue_size=max_queue_size,
    ):
        yield {node_order[index] for index in subgraph}

    if allow_zero_outputs:
        reversed_graph = graph.reverse(copy=True)
        reversed_payload, reversed_node_order = graph_to_input(
            reversed_graph,
            weighted=weighted,
            weight_attr=weight_attr,
            forbidden_attr=forbidden_attr,
            forbid_sources_and_sinks=forbid_sources_and_sinks,
            ordering=ordering,
        )
        for subgraph in iter_all_graph_input(
            reversed_payload,
            0,
            max_num_inputs,
            max_queue_size=max_queue_size,
        ):
            yield {reversed_node_order[index] for index in subgraph}
