from __future__ import annotations

from collections.abc import Hashable, Iterator
from typing import Literal, TypeVar

import networkx as nx

from ._native import GraphInput, iter_all_graph_input, solve_graph_input


NodeT = TypeVar("NodeT", bound=Hashable)


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
    iteration_type: str = "linear-rev",
    ordering: Literal["default", "sort", "toposort"] = "toposort",
    flags: int = 0xFF,
) -> Iterator[set[NodeT]]:
    """
    Enumerate the maximum directed convex subgraphs of a DAG under I/O constraints.

    This algorithm is the implementation for Giaquinta et. al. "Maximum Convex Subgraphs Under I/O Constraint for
    Automatic Identification of Custom Instructions"
    """

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
    return (
        {node_order[index] for index in subgraph}
        for subgraph in iter_all_graph_input(
            payload,
            max_num_inputs,
            max_num_outputs,
            max_queue_size=max_queue_size,
        )
    )
