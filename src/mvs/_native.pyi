from __future__ import annotations


class GraphInput:
    name: str
    num_nodes: int
    edges: list[tuple[int, int]]
    alternate_edges: list[tuple[int, int]]
    weights: list[float]
    forbidden: list[int]
    frequency: int
    forbid_sources_and_sinks: bool


class SolveResult:
    max_weight: float
    subgraphs: list[list[int]]


class ExhaustiveSubgraphIterator:
    def __iter__(self) -> ExhaustiveSubgraphIterator: ...
    def __next__(self) -> list[int]: ...
    def close(self) -> None: ...


def solve_graph_input(
    graph_input: GraphInput,
    max_num_inputs: int,
    max_num_outputs: int,
    max_subgraph_size: int = -1,
    iteration_type: str = "linear-rev",
    flags: int = 0xFF,
) -> SolveResult: ...


def solve_all_graph_input(
    graph_input: GraphInput,
    max_num_inputs: int,
    max_num_outputs: int,
    max_subgraph_size: int = -1,
    connected_only: bool = False,
) -> SolveResult: ...


def iter_all_graph_input(
    graph_input: GraphInput,
    max_num_inputs: int,
    max_num_outputs: int,
    max_subgraph_size: int = -1,
    max_queue_size: int = 128,
    connected_only: bool = False,
) -> ExhaustiveSubgraphIterator: ...
