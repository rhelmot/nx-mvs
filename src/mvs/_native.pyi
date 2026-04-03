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


def solve_graph_input(
    graph_input: GraphInput,
    max_num_inputs: int,
    max_num_outputs: int,
    iteration_type: str = "linear-rev",
    flags: int = 0xFF,
) -> SolveResult: ...
