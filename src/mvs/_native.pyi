from __future__ import annotations


class GraphInput:
    name: str
    num_nodes: int
    edges: list[tuple[int, int]]
    weights: list[float]
    forbidden: list[int]
    frequency: int


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
    iteration_type: str = "linear-rev",
    flags: int = 0xFF,
) -> SolveResult: ...


def solve_all_graph_input(
    graph_input: GraphInput,
    max_num_inputs: int,
    max_num_outputs: int,
) -> SolveResult: ...


def iter_all_graph_input(
    graph_input: GraphInput,
    max_num_inputs: int,
    max_num_outputs: int,
    max_queue_size: int = 128,
) -> ExhaustiveSubgraphIterator: ...
