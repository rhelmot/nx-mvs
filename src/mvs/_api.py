from __future__ import annotations

from collections.abc import Callable, Hashable, Iterator
from dataclasses import dataclass, replace
import math
from typing import Generic, Literal, TypeVar, cast

import networkx as nx

from ._native import (
    GraphInput,
    grow_nonzero_output_graph_input,
    grow_zero_output_graph_input,
    iter_all_graph_input,
    sample_nonzero_output_graph_input,
    sample_zero_output_graph_input,
    solve_graph_input,
)


NodeT = TypeVar("NodeT", bound=Hashable)
Ordering = Literal["default", "sort", "toposort"]

_AUTO_SAMPLING_RESULT_THRESHOLD = 10_000
_AUTO_EXHAUSTIVE_WORK_THRESHOLD = 65_536
_AUTO_ALTERNATE_ONLY_EXHAUSTIVE_WORK_THRESHOLD = 4_096
_AUTO_ALTERNATE_ONLY_SMALL_GRAPH_NODES = 512


class _ExhaustiveWorkLimitReached(Exception):
    pass


@dataclass(frozen=True, slots=True)
class ConvexSubgraphQuery(Generic[NodeT]):
    max_subgraph_size: int | None = None
    weighted: bool = False
    weight_attr: str = "weight"
    forbidden_attr: str | None = "forbidden"
    body_forbidden_attr: str | None = None
    input_forbidden_attr: str | None = None
    forbid_sources_and_sinks: bool = True
    connected_only: bool = False
    alternate_connected_only: bool = False
    max_queue_size: int = 128
    iteration_type: str = "linear-rev"
    flags: int = 0xFF
    ordering: Ordering = "toposort"
    sampling_max_states_expanded: int = 10000
    sampling_max_samples: int = 1000
    sampling_max_children_per_state: int = 2
    sampling_size_bin_width: int = 4
    sampling_thicken_radius: int = 1
    sampling_bucket_by_num_inputs: bool = True
    sampling_bucket_by_num_outputs: bool = True
    sampling_minimal_node_bin_width: int = 1
    sampling_boundary_pair_samples: int = 512
    sampling_passes: int = 1
    sampling_exact_kernel_size: int = 0

    def enumerate(
        self,
        graph: nx.DiGraph[NodeT],
        max_num_inputs: int,
        max_num_outputs: int = 1,
        *,
        alternate_graph: nx.DiGraph[NodeT] | None = None,
        allow_zero_outputs: bool = False,
        sampling: bool | None = None,
    ) -> Iterator[set[NodeT]]:
        return self._operation(
            graph,
            max_num_inputs,
            max_num_outputs,
            alternate_graph=alternate_graph,
            allow_zero_outputs=allow_zero_outputs,
        ).enumerate(sampling=sampling)

    def sample(
        self,
        graph: nx.DiGraph[NodeT],
        max_num_inputs: int,
        max_num_outputs: int = 1,
        *,
        alternate_graph: nx.DiGraph[NodeT] | None = None,
        allow_zero_outputs: bool = False,
    ) -> Iterator[set[NodeT]]:
        return self._operation(
            graph,
            max_num_inputs,
            max_num_outputs,
            alternate_graph=alternate_graph,
            allow_zero_outputs=allow_zero_outputs,
        ).sample()

    def maximum(
        self,
        graph: nx.DiGraph[NodeT],
        max_num_inputs: int,
        max_num_outputs: int = 1,
        *,
        alternate_graph: nx.DiGraph[NodeT] | None = None,
        allow_zero_outputs: bool = False,
        sampling: bool | None = None,
    ) -> Iterator[set[NodeT]]:
        return self._operation(
            graph,
            max_num_inputs,
            max_num_outputs,
            alternate_graph=alternate_graph,
            allow_zero_outputs=allow_zero_outputs,
        ).maximum(
            sampling=sampling,
        )

    def grow(
        self,
        graph: nx.DiGraph[NodeT],
        seed_nodes: set[NodeT],
        *,
        max_num_inputs: int = 4,
        max_num_outputs: int = 1,
        alternate_graph: nx.DiGraph[NodeT] | None = None,
        allow_zero_outputs: bool = False,
        oracle: Callable[..., object | None] | None = None,
        initial_oracle_state: object | None = None,
    ) -> Iterator[set[NodeT]]:
        return self._operation(
            graph,
            max_num_inputs,
            max_num_outputs,
            alternate_graph=alternate_graph,
            allow_zero_outputs=allow_zero_outputs,
        ).grow(
            seed_nodes,
            oracle=oracle,
            initial_oracle_state=initial_oracle_state,
        )

    def _operation(
        self,
        graph: nx.DiGraph[NodeT],
        max_num_inputs: int,
        max_num_outputs: int,
        *,
        alternate_graph: nx.DiGraph[NodeT] | None,
        allow_zero_outputs: bool,
    ) -> _ConvexSubgraphOperation[NodeT]:
        return _ConvexSubgraphOperation(
            graph,
            max_num_inputs=max_num_inputs,
            max_num_outputs=max_num_outputs,
            alternate_graph=alternate_graph,
            max_subgraph_size=self.max_subgraph_size,
            weighted=self.weighted,
            weight_attr=self.weight_attr,
            forbidden_attr=self.forbidden_attr,
            body_forbidden_attr=self.body_forbidden_attr,
            input_forbidden_attr=self.input_forbidden_attr,
            forbid_sources_and_sinks=self.forbid_sources_and_sinks,
            allow_zero_outputs=allow_zero_outputs,
            connected_only=self.connected_only,
            alternate_connected_only=self.alternate_connected_only,
            max_queue_size=self.max_queue_size,
            iteration_type=self.iteration_type,
            flags=self.flags,
            ordering=self.ordering,
            sampling_max_states_expanded=self.sampling_max_states_expanded,
            sampling_max_samples=self.sampling_max_samples,
            sampling_max_children_per_state=self.sampling_max_children_per_state,
            sampling_size_bin_width=self.sampling_size_bin_width,
            sampling_thicken_radius=self.sampling_thicken_radius,
            sampling_bucket_by_num_inputs=self.sampling_bucket_by_num_inputs,
            sampling_bucket_by_num_outputs=self.sampling_bucket_by_num_outputs,
            sampling_minimal_node_bin_width=self.sampling_minimal_node_bin_width,
            sampling_boundary_pair_samples=self.sampling_boundary_pair_samples,
            sampling_passes=self.sampling_passes,
            sampling_exact_kernel_size=self.sampling_exact_kernel_size,
        )


@dataclass(frozen=True, slots=True)
class _ConvexSubgraphOperation(Generic[NodeT]):
    graph: nx.DiGraph[NodeT]
    max_num_inputs: int
    max_num_outputs: int = 1
    alternate_graph: nx.DiGraph[NodeT] | None = None
    max_subgraph_size: int | None = None
    weighted: bool = False
    weight_attr: str = "weight"
    forbidden_attr: str | None = "forbidden"
    body_forbidden_attr: str | None = None
    input_forbidden_attr: str | None = None
    forbid_sources_and_sinks: bool = True
    allow_zero_outputs: bool = False
    connected_only: bool = False
    alternate_connected_only: bool = False
    max_queue_size: int = 128
    iteration_type: str = "linear-rev"
    flags: int = 0xFF
    ordering: Ordering = "toposort"
    sampling_max_states_expanded: int = 10000
    sampling_max_samples: int = 1000
    sampling_max_children_per_state: int = 2
    sampling_size_bin_width: int = 4
    sampling_thicken_radius: int = 1
    sampling_bucket_by_num_inputs: bool = True
    sampling_bucket_by_num_outputs: bool = True
    sampling_minimal_node_bin_width: int = 1
    sampling_boundary_pair_samples: int = 512
    sampling_passes: int = 1
    sampling_exact_kernel_size: int = 0

    def enumerate(
        self,
        *,
        sampling: bool | None = None,
        allow_zero_outputs: bool | None = None,
        connected_only: bool | None = None,
        alternate_connected_only: bool | None = None,
    ) -> Iterator[set[NodeT]]:
        return self._enumerate_convex_subgraphs(
            allow_zero_outputs=allow_zero_outputs,
            connected_only=connected_only,
            alternate_connected_only=alternate_connected_only,
            ordering=self.ordering,
            sampling=sampling,
        )

    def sample(
        self,
        *,
        allow_zero_outputs: bool | None = None,
    ) -> Iterator[set[NodeT]]:
        include_zero_outputs = (
            self.allow_zero_outputs
            if allow_zero_outputs is None
            else allow_zero_outputs
        )
        if self.max_num_outputs == 0:
            yield from self._sample_zero_output()
            return
        if not include_zero_outputs:
            yield from self._sample_nonzero_output()
            return

        seen: set[frozenset[NodeT]] = set()
        for source in (
            self._sample_nonzero_output(),
            replace(self, max_num_outputs=0)._sample_zero_output(),
        ):
            for subgraph in source:
                key = frozenset(subgraph)
                if key in seen:
                    continue
                seen.add(key)
                yield subgraph

    def maximum(
        self,
        *,
        sampling: bool | None = None,
        allow_zero_outputs: bool | None = None,
        connected_only: bool | None = None,
        alternate_connected_only: bool | None = None,
    ) -> Iterator[set[NodeT]]:
        return self._enumerate_maximum_convex_subgraphs(
            allow_zero_outputs=allow_zero_outputs,
            connected_only=connected_only,
            alternate_connected_only=alternate_connected_only,
            ordering=self.ordering,
            sampling=sampling,
        )

    def grow(
        self,
        seed_nodes: set[NodeT],
        *,
        oracle: Callable[..., object | None] | None = None,
        initial_oracle_state: object | None = None,
    ) -> Iterator[set[NodeT]]:
        if self.max_num_outputs == 0:
            return self._grow_zero_output(
                seed_nodes,
                oracle=oracle,
                initial_oracle_state=initial_oracle_state,
            )
        return self._grow_nonzero_output(
            seed_nodes,
            oracle=oracle,
            initial_oracle_state=initial_oracle_state,
        )

    def _native_max_subgraph_size(self) -> int:
        native_max_subgraph_size = (
            -1 if self.max_subgraph_size is None else self.max_subgraph_size
        )
        if native_max_subgraph_size < -1:
            raise ValueError("max_subgraph_size must be non-negative or None")
        return native_max_subgraph_size

    def _validate_sampling_parameters(self, *, nonzero: bool = False) -> None:
        if self.sampling_max_states_expanded < 0 or self.sampling_max_samples < 0:
            raise ValueError("sampling budgets must be non-negative")
        if self.sampling_max_children_per_state <= 0:
            raise ValueError("sampling_max_children_per_state must be positive")
        if self.sampling_size_bin_width <= 0:
            raise ValueError("sampling_size_bin_width must be positive")
        if self.sampling_thicken_radius < 0:
            raise ValueError("sampling_thicken_radius must be non-negative")
        if self.sampling_minimal_node_bin_width < 0:
            raise ValueError("sampling_minimal_node_bin_width must be non-negative")
        if nonzero and self.sampling_boundary_pair_samples < 0:
            raise ValueError("sampling_boundary_pair_samples must be non-negative")
        if self.sampling_passes <= 0:
            raise ValueError("sampling_passes must be positive")
        if self.sampling_exact_kernel_size < 0:
            raise ValueError("sampling_exact_kernel_size must be non-negative")

    def _validate_alternate_connected_only(
        self,
        alternate_connected_only: bool,
    ) -> None:
        if alternate_connected_only and self.alternate_graph is None:
            raise ValueError(
                "alternate_connected_only requires alternate_graph"
            )

    def _auto_exhaustive_work_limit(
        self,
        *,
        connected_only: bool,
        alternate_connected_only: bool,
    ) -> int:
        if connected_only:
            return _AUTO_EXHAUSTIVE_WORK_THRESHOLD
        if not alternate_connected_only:
            return 0

        node_count = self.graph.number_of_nodes()
        if self.alternate_graph is not None:
            node_count = max(node_count, self.alternate_graph.number_of_nodes())
        if node_count <= _AUTO_ALTERNATE_ONLY_SMALL_GRAPH_NODES:
            return _AUTO_EXHAUSTIVE_WORK_THRESHOLD
        return _AUTO_ALTERNATE_ONLY_EXHAUSTIVE_WORK_THRESHOLD

    def _enumerate_maximum_convex_subgraphs(
        self,
        *,
        allow_zero_outputs: bool | None = None,
        connected_only: bool | None = None,
        alternate_connected_only: bool | None = None,
        ordering: Ordering = "toposort",
        sampling: bool | None = None,
    ) -> Iterator[set[NodeT]]:
        native_max_subgraph_size = self._native_max_subgraph_size()
        sampling = _validate_sampling(sampling)
        allow_zero_outputs = (
            self.allow_zero_outputs
            if allow_zero_outputs is None
            else allow_zero_outputs
        )
        connected_only = (
            self.connected_only if connected_only is None else connected_only
        )
        alternate_connected_only = (
            self.alternate_connected_only
            if alternate_connected_only is None
            else alternate_connected_only
        )
        self._validate_alternate_connected_only(alternate_connected_only)

        if sampling is True:
            return self._iter_sampled_maximum_convex_subgraphs(
                allow_zero_outputs=allow_zero_outputs,
                connected_only=connected_only,
                alternate_connected_only=alternate_connected_only,
                ordering=ordering,
            )

        if self.max_num_outputs <= 1:
            best_weight = float("-inf")
            best_subgraphs: list[set[NodeT]] = []
            for subgraph in self._enumerate_convex_subgraphs(
                allow_zero_outputs=allow_zero_outputs,
                connected_only=connected_only,
                alternate_connected_only=alternate_connected_only,
                ordering=ordering,
                sampling=sampling,
            ):
                weight = _subgraph_weight(
                    self.graph,
                    subgraph,
                    weighted=self.weighted,
                    weight_attr=self.weight_attr,
                )
                if weight > best_weight:
                    best_weight = weight
                    best_subgraphs = [subgraph]
                elif math.isclose(weight, best_weight):
                    best_subgraphs.append(subgraph)
            return iter(best_subgraphs)

        if self.alternate_graph is not None:
            raise NotImplementedError(
                "alternate_graph is currently only supported for exhaustive enumeration "
                "and maximum enumeration when max_num_outputs <= 1"
            )

        if connected_only:
            best_weight = float("-inf")
            best_subgraphs: list[set[NodeT]] = []
            for component_nodes in _connected_component_node_sets(
                self.graph,
                self.alternate_graph,
            ):
                component = cast(
                    "nx.DiGraph[NodeT]",
                    self.graph.subgraph(component_nodes).copy(),
                )
                payload, node_order = graph_to_input(
                    component,
                    alternate_graph=(
                        cast(
                            "nx.DiGraph[NodeT]",
                            self.alternate_graph.subgraph(component_nodes).copy(),
                        )
                        if self.alternate_graph is not None
                        else None
                    ),
                    weighted=self.weighted,
                    weight_attr=self.weight_attr,
                    forbidden_attr=self.forbidden_attr,
                    body_forbidden_attr=self.body_forbidden_attr,
                    input_forbidden_attr=self.input_forbidden_attr,
                    forbid_sources_and_sinks=self.forbid_sources_and_sinks,
                    ordering=ordering,
                )
                result = solve_graph_input(
                    payload,
                    self.max_num_inputs,
                    self.max_num_outputs,
                    native_max_subgraph_size,
                    iteration_type=self.iteration_type,
                    flags=self.flags,
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
            self.graph,
            alternate_graph=self.alternate_graph,
            weighted=self.weighted,
            weight_attr=self.weight_attr,
            forbidden_attr=self.forbidden_attr,
            body_forbidden_attr=self.body_forbidden_attr,
            input_forbidden_attr=self.input_forbidden_attr,
            forbid_sources_and_sinks=self.forbid_sources_and_sinks,
            ordering=ordering,
        )
        result = solve_graph_input(
            payload,
            self.max_num_inputs,
            self.max_num_outputs,
            native_max_subgraph_size,
            iteration_type=self.iteration_type,
            flags=self.flags,
        )
        return ({node_order[index] for index in subgraph} for subgraph in result.subgraphs)

    def _enumerate_convex_subgraphs(
        self,
        *,
        allow_zero_outputs: bool | None = None,
        connected_only: bool | None = None,
        alternate_connected_only: bool | None = None,
        ordering: Ordering | None = None,
        sampling: bool | None = None,
    ) -> Iterator[set[NodeT]]:
        native_max_subgraph_size = self._native_max_subgraph_size()
        allow_zero_outputs = (
            self.allow_zero_outputs
            if allow_zero_outputs is None
            else allow_zero_outputs
        )
        connected_only = (
            self.connected_only if connected_only is None else connected_only
        )
        alternate_connected_only = (
            self.alternate_connected_only
            if alternate_connected_only is None
            else alternate_connected_only
        )
        self._validate_alternate_connected_only(alternate_connected_only)
        ordering = self.ordering if ordering is None else ordering
        sampling = _validate_sampling(sampling)

        def iter_exhaustive_results(
            max_work: int | None = None,
        ) -> Iterator[set[NodeT]]:
            native_max_work = 0 if max_work is None else max_work
            if connected_only or alternate_connected_only:
                for component_nodes in _connected_component_node_sets(
                    self.graph,
                    self.alternate_graph,
                ):
                    component = cast(
                        "nx.DiGraph[NodeT]",
                        self.graph.subgraph(component_nodes).copy(),
                    )
                    component_alternate = (
                        cast(
                            "nx.DiGraph[NodeT]",
                            self.alternate_graph.subgraph(component_nodes).copy(),
                        )
                        if self.alternate_graph is not None
                        else None
                    )
                    payload, node_order = graph_to_input(
                        component,
                        alternate_graph=component_alternate,
                        weighted=self.weighted,
                        weight_attr=self.weight_attr,
                        forbidden_attr=self.forbidden_attr,
                        body_forbidden_attr=self.body_forbidden_attr,
                        input_forbidden_attr=self.input_forbidden_attr,
                        forbid_sources_and_sinks=self.forbid_sources_and_sinks,
                        ordering=ordering,
                    )
                    yield from _iter_convex_subgraphs(
                        component,
                        component_alternate,
                        payload,
                        node_order,
                        max_num_inputs=self.max_num_inputs,
                        max_num_outputs=self.max_num_outputs,
                        max_subgraph_size=native_max_subgraph_size,
                        weighted=self.weighted,
                        weight_attr=self.weight_attr,
                        forbidden_attr=self.forbidden_attr,
                        forbid_sources_and_sinks=self.forbid_sources_and_sinks,
                        allow_zero_outputs=allow_zero_outputs,
                        connected_only=connected_only,
                        alternate_connected_only=alternate_connected_only,
                        ordering=ordering,
                        max_queue_size=self.max_queue_size,
                        max_work=native_max_work,
                    )
                return

            payload, node_order = graph_to_input(
                self.graph,
                alternate_graph=self.alternate_graph,
                weighted=self.weighted,
                weight_attr=self.weight_attr,
                forbidden_attr=self.forbidden_attr,
                body_forbidden_attr=self.body_forbidden_attr,
                input_forbidden_attr=self.input_forbidden_attr,
                forbid_sources_and_sinks=self.forbid_sources_and_sinks,
                ordering=ordering,
            )
            yield from _iter_convex_subgraphs(
                self.graph,
                self.alternate_graph,
                payload,
                node_order,
                max_num_inputs=self.max_num_inputs,
                max_num_outputs=self.max_num_outputs,
                max_subgraph_size=native_max_subgraph_size,
                weighted=self.weighted,
                weight_attr=self.weight_attr,
                forbidden_attr=self.forbidden_attr,
                forbid_sources_and_sinks=self.forbid_sources_and_sinks,
                allow_zero_outputs=allow_zero_outputs,
                connected_only=connected_only,
                alternate_connected_only=alternate_connected_only,
                ordering=ordering,
                max_queue_size=self.max_queue_size,
                max_work=native_max_work,
            )

        def iter_sampled_results() -> Iterator[set[NodeT]]:
            yield from self._iter_sampled_convex_subgraphs(
                allow_zero_outputs=allow_zero_outputs,
                connected_only=connected_only,
                alternate_connected_only=alternate_connected_only,
                ordering=ordering,
            )

        if sampling is False:
            return iter_exhaustive_results()
        if sampling is True:
            return iter_sampled_results()
        return _iter_exact_then_sampled(
            iter_exhaustive_results,
            iter_sampled_results,
            threshold=_AUTO_SAMPLING_RESULT_THRESHOLD,
            work_limit=self._auto_exhaustive_work_limit(
                connected_only=connected_only,
                alternate_connected_only=alternate_connected_only,
            ),
        )

    def _sample_zero_output(self) -> Iterator[set[NodeT]]:
        native_max_subgraph_size = self._native_max_subgraph_size()
        self._validate_sampling_parameters()

        pass_configs = _sampling_pass_configs(
            self.ordering,
            max_children_per_state=self.sampling_max_children_per_state,
            size_bin_width=self.sampling_size_bin_width,
            pass_count=self.sampling_passes,
        )
        per_pass_states = (
            0 if self.sampling_max_states_expanded == 0
            else max(1, math.ceil(self.sampling_max_states_expanded / len(pass_configs)))
        )

        for component_nodes in _connected_component_node_sets(
            self.graph,
            self.alternate_graph,
        ):
            component = cast(
                "nx.DiGraph[NodeT]",
                self.graph.subgraph(component_nodes).copy(),
            )
            component_alternate = (
                cast(
                    "nx.DiGraph[NodeT]",
                    self.alternate_graph.subgraph(component_nodes).copy(),
                )
                if self.alternate_graph is not None
                else None
            )
            seen: set[frozenset[NodeT]] = set()
            emitted_count = 0

            if self.sampling_exact_kernel_size > 0:
                exact_limit = (
                    self.sampling_exact_kernel_size
                    if native_max_subgraph_size < 0
                    else min(native_max_subgraph_size, self.sampling_exact_kernel_size)
                )
                exact_query = replace(
                    self,
                    graph=component,
                    max_num_outputs=0,
                    alternate_graph=component_alternate,
                    max_subgraph_size=exact_limit,
                    allow_zero_outputs=False,
                    connected_only=True,
                )
                for subgraph in exact_query._enumerate_convex_subgraphs(
                    allow_zero_outputs=False,
                    connected_only=True,
                    sampling=False,
                ):
                    key = frozenset(subgraph)
                    if key in seen:
                        continue
                    seen.add(key)
                    emitted_count += 1
                    yield subgraph
                    if (
                        self.sampling_max_samples > 0
                        and emitted_count >= self.sampling_max_samples
                    ):
                        return

            for pass_index, (
                pass_ordering,
                pass_children,
                pass_size_bin_width,
            ) in enumerate(pass_configs):
                if self.sampling_max_samples <= 0:
                    return
                remaining_samples = self.sampling_max_samples - emitted_count
                if remaining_samples <= 0:
                    return
                remaining_passes = len(pass_configs) - pass_index
                pass_samples = max(
                    1,
                    math.ceil(remaining_samples / remaining_passes),
                )
                pass_states = (
                    0 if self.sampling_max_states_expanded == 0
                    else min(
                        per_pass_states,
                        max(
                            1,
                            math.ceil(
                                per_pass_states
                                * remaining_samples
                                / self.sampling_max_samples
                            ),
                        ),
                    )
                )
                payload, node_order = graph_to_input(
                    component,
                    alternate_graph=component_alternate,
                    weighted=self.weighted,
                    weight_attr=self.weight_attr,
                    forbidden_attr=self.forbidden_attr,
                    body_forbidden_attr=self.body_forbidden_attr,
                    input_forbidden_attr=self.input_forbidden_attr,
                    forbid_sources_and_sinks=self.forbid_sources_and_sinks,
                    ordering=pass_ordering,
                )
                result = sample_zero_output_graph_input(
                    payload,
                    self.max_num_inputs,
                    native_max_subgraph_size,
                    max_states_expanded=pass_states,
                    max_samples=pass_samples,
                    max_children_per_state=pass_children,
                    size_bin_width=pass_size_bin_width,
                    thicken_radius=self.sampling_thicken_radius,
                    bucket_by_num_inputs=self.sampling_bucket_by_num_inputs,
                    minimal_node_bin_width=self.sampling_minimal_node_bin_width,
                )
                for subgraph in result.subgraphs:
                    nodes = {node_order[index] for index in subgraph}
                    if (
                        self.sampling_exact_kernel_size > 0
                        and len(nodes) <= self.sampling_exact_kernel_size
                    ):
                        continue
                    key = frozenset(nodes)
                    if key in seen:
                        continue
                    seen.add(key)
                    emitted_count += 1
                    yield nodes
                    if (
                        self.sampling_max_samples > 0
                        and emitted_count >= self.sampling_max_samples
                    ):
                        return

    def _sample_nonzero_output(self) -> Iterator[set[NodeT]]:
        native_max_subgraph_size = self._native_max_subgraph_size()
        if self.max_num_outputs <= 0:
            raise ValueError("max_num_outputs must be positive")
        self._validate_sampling_parameters(nonzero=True)

        pass_configs = _sampling_pass_configs(
            self.ordering,
            max_children_per_state=self.sampling_max_children_per_state,
            size_bin_width=self.sampling_size_bin_width,
            pass_count=self.sampling_passes,
        )
        per_pass_states = (
            0 if self.sampling_max_states_expanded == 0
            else max(1, math.ceil(self.sampling_max_states_expanded / len(pass_configs)))
        )

        for component_nodes in _connected_component_node_sets(
            self.graph,
            self.alternate_graph,
        ):
            component = cast(
                "nx.DiGraph[NodeT]",
                self.graph.subgraph(component_nodes).copy(),
            )
            component_alternate = (
                cast(
                    "nx.DiGraph[NodeT]",
                    self.alternate_graph.subgraph(component_nodes).copy(),
                )
                if self.alternate_graph is not None
                else None
            )
            seen: set[frozenset[NodeT]] = set()
            emitted_count = 0

            if self.sampling_exact_kernel_size > 0:
                exact_limit = (
                    self.sampling_exact_kernel_size
                    if native_max_subgraph_size < 0
                    else min(native_max_subgraph_size, self.sampling_exact_kernel_size)
                )
                exact_query = replace(
                    self,
                    graph=component,
                    alternate_graph=component_alternate,
                    max_subgraph_size=exact_limit,
                    allow_zero_outputs=False,
                    connected_only=True,
                )
                for subgraph in exact_query._enumerate_convex_subgraphs(
                    allow_zero_outputs=False,
                    connected_only=True,
                    sampling=False,
                ):
                    key = frozenset(subgraph)
                    if key in seen:
                        continue
                    seen.add(key)
                    emitted_count += 1
                    yield subgraph
                    if (
                        self.sampling_max_samples > 0
                        and emitted_count >= self.sampling_max_samples
                    ):
                        return

            for pass_index, (
                pass_ordering,
                pass_children,
                pass_size_bin_width,
            ) in enumerate(pass_configs):
                if self.sampling_max_samples <= 0:
                    return
                remaining_samples = self.sampling_max_samples - emitted_count
                if remaining_samples <= 0:
                    return
                remaining_passes = len(pass_configs) - pass_index
                pass_samples = max(
                    1,
                    math.ceil(remaining_samples / remaining_passes),
                )
                pass_states = (
                    0 if self.sampling_max_states_expanded == 0
                    else min(
                        per_pass_states,
                        max(
                            1,
                            math.ceil(
                                per_pass_states
                                * remaining_samples
                                / self.sampling_max_samples
                            ),
                        ),
                    )
                )
                pass_boundary_pair_samples = (
                    self.sampling_boundary_pair_samples
                    if pass_index == 0
                    else min(self.sampling_boundary_pair_samples, 32)
                )
                payload, node_order = graph_to_input(
                    component,
                    alternate_graph=component_alternate,
                    weighted=self.weighted,
                    weight_attr=self.weight_attr,
                    forbidden_attr=self.forbidden_attr,
                    body_forbidden_attr=self.body_forbidden_attr,
                    input_forbidden_attr=self.input_forbidden_attr,
                    forbid_sources_and_sinks=self.forbid_sources_and_sinks,
                    ordering=pass_ordering,
                )
                result = sample_nonzero_output_graph_input(
                    payload,
                    self.max_num_inputs,
                    self.max_num_outputs,
                    native_max_subgraph_size,
                    max_states_expanded=pass_states,
                    max_samples=pass_samples,
                    max_children_per_state=pass_children,
                    size_bin_width=pass_size_bin_width,
                    thicken_radius=self.sampling_thicken_radius,
                    bucket_by_num_inputs=self.sampling_bucket_by_num_inputs,
                    bucket_by_num_outputs=self.sampling_bucket_by_num_outputs,
                    minimal_node_bin_width=self.sampling_minimal_node_bin_width,
                    boundary_pair_samples=pass_boundary_pair_samples,
                )
                for subgraph in result.subgraphs:
                    nodes = {node_order[index] for index in subgraph}
                    if (
                        self.sampling_exact_kernel_size > 0
                        and len(nodes) <= self.sampling_exact_kernel_size
                    ):
                        continue
                    key = frozenset(nodes)
                    if key in seen:
                        continue
                    seen.add(key)
                    emitted_count += 1
                    yield nodes
                    if (
                        self.sampling_max_samples > 0
                        and emitted_count >= self.sampling_max_samples
                    ):
                        return

    def _grow_zero_output(
        self,
        seed_nodes: set[NodeT],
        *,
        oracle: Callable[..., object | None] | None = None,
        initial_oracle_state: object | None = None,
    ) -> Iterator[set[NodeT]]:
        native_max_subgraph_size = self._native_max_subgraph_size()
        self._validate_alternate_connected_only(self.alternate_connected_only)
        payload, node_order = graph_to_input(
            self.graph,
            alternate_graph=self.alternate_graph,
            forbidden_attr=self.forbidden_attr,
            body_forbidden_attr=self.body_forbidden_attr,
            input_forbidden_attr=self.input_forbidden_attr,
            forbid_sources_and_sinks=self.forbid_sources_and_sinks,
            ordering=self.ordering,
        )
        node_index = {node: index for index, node in enumerate(node_order)}
        seed_indices: list[int] = []
        for node in seed_nodes:
            try:
                seed_indices.append(node_index[node])
            except KeyError as exc:
                raise ValueError(
                    f"seed node {node!r} is not present in the graph set"
                ) from exc

        def oracle_indices(state: object | None, indices: list[int]) -> object | None:
            nodes = {node_order[index] for index in indices}
            assert oracle is not None
            return oracle(state, nodes)

        result = grow_zero_output_graph_input(
            payload,
            seed_indices,
            self.max_num_inputs,
            native_max_subgraph_size,
            oracle_indices if oracle is not None else None,
            initial_oracle_state,
        )
        return (
            nodes
            for nodes in (
                {node_order[index] for index in subgraph}
                for subgraph in result.subgraphs
            )
            if _passes_alternate_connectivity(
                self.alternate_graph,
                nodes,
                alternate_connected_only=self.alternate_connected_only,
            )
        )

    def _grow_nonzero_output(
        self,
        seed_nodes: set[NodeT],
        *,
        oracle: Callable[..., object | None] | None = None,
        initial_oracle_state: object | None = None,
    ) -> Iterator[set[NodeT]]:
        native_max_subgraph_size = self._native_max_subgraph_size()
        self._validate_alternate_connected_only(self.alternate_connected_only)
        if self.max_num_outputs <= 0:
            raise ValueError("max_num_outputs must be positive")

        payload, node_order = graph_to_input(
            self.graph,
            alternate_graph=self.alternate_graph,
            forbidden_attr=self.forbidden_attr,
            body_forbidden_attr=self.body_forbidden_attr,
            input_forbidden_attr=self.input_forbidden_attr,
            forbid_sources_and_sinks=self.forbid_sources_and_sinks,
            ordering=self.ordering,
        )
        node_index = {node: index for index, node in enumerate(node_order)}
        seed_indices: list[int] = []
        for node in seed_nodes:
            try:
                seed_indices.append(node_index[node])
            except KeyError as exc:
                raise ValueError(
                    f"seed node {node!r} is not present in the graph set"
                ) from exc

        def oracle_indices(state: object | None, indices: list[int]) -> object | None:
            nodes = {node_order[index] for index in indices}
            assert oracle is not None
            return oracle(state, nodes)

        result = grow_nonzero_output_graph_input(
            payload,
            seed_indices,
            self.max_num_inputs,
            self.max_num_outputs,
            native_max_subgraph_size,
            oracle_indices if oracle is not None else None,
            initial_oracle_state,
        )
        return (
            nodes
            for nodes in (
                {node_order[index] for index in subgraph}
                for subgraph in result.subgraphs
            )
            if _passes_alternate_connectivity(
                self.alternate_graph,
                nodes,
                alternate_connected_only=self.alternate_connected_only,
            )
        )

    def _iter_sampled_convex_subgraphs(
        self,
        *,
        allow_zero_outputs: bool | None = None,
        connected_only: bool | None = None,
        alternate_connected_only: bool | None = None,
        ordering: Ordering | None = None,
    ) -> Iterator[set[NodeT]]:
        allow_zero_outputs = (
            self.allow_zero_outputs
            if allow_zero_outputs is None
            else allow_zero_outputs
        )
        connected_only = (
            self.connected_only if connected_only is None else connected_only
        )
        alternate_connected_only = (
            self.alternate_connected_only
            if alternate_connected_only is None
            else alternate_connected_only
        )
        self._validate_alternate_connected_only(alternate_connected_only)
        ordering = self.ordering if ordering is None else ordering

        if self.max_num_inputs < 0 or self.max_num_outputs < 0:
            raise ValueError("I/O limits must be non-negative")

        seen: set[frozenset[NodeT]] = set()

        if self.max_num_outputs > 0:
            for subgraph in self._sample_nonzero_output():
                if not _passes_alternate_connectivity(
                    self.alternate_graph,
                    subgraph,
                    alternate_connected_only=alternate_connected_only,
                ):
                    continue
                key = frozenset(subgraph)
                if key in seen:
                    continue
                seen.add(key)
                yield subgraph

        if self.max_num_outputs == 0 or allow_zero_outputs:
            for subgraph in replace(self, max_num_outputs=0)._sample_zero_output():
                if not _passes_alternate_connectivity(
                    self.alternate_graph,
                    subgraph,
                    alternate_connected_only=alternate_connected_only,
                ):
                    continue
                key = frozenset(subgraph)
                if key in seen:
                    continue
                seen.add(key)
                yield subgraph

    def _iter_sampled_maximum_convex_subgraphs(
        self,
        *,
        allow_zero_outputs: bool | None = None,
        connected_only: bool | None = None,
        alternate_connected_only: bool | None = None,
        ordering: Ordering | None = None,
    ) -> Iterator[set[NodeT]]:
        best_weight = float("-inf")
        best_subgraphs: list[set[NodeT]] = []
        for subgraph in self._iter_sampled_convex_subgraphs(
            allow_zero_outputs=allow_zero_outputs,
            connected_only=connected_only,
            alternate_connected_only=alternate_connected_only,
            ordering=ordering,
        ):
            weight = _subgraph_weight(
                self.graph,
                subgraph,
                weighted=self.weighted,
                weight_attr=self.weight_attr,
            )
            if weight > best_weight:
                best_weight = weight
                best_subgraphs = [subgraph]
            elif math.isclose(weight, best_weight):
                best_subgraphs.append(subgraph)

        yield from best_subgraphs


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


def _validate_sampling(value: bool | None) -> bool | None:
    if value is not None and not isinstance(value, bool):
        raise TypeError("sampling must be True, False, or None")
    return value


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


def _num_outputs(
    graph: nx.DiGraph[NodeT],
    subgraph: set[NodeT],
) -> int:
    return sum(
        1
        for node in subgraph
        if any(successor not in subgraph for successor in graph.successors(node))
    )


def _is_connected_with_inputs(
    graph: nx.DiGraph[NodeT],
    subgraph: set[NodeT],
) -> bool:
    inputs = {
        predecessor
        for node in subgraph
        if node in graph
        for predecessor in graph.predecessors(node)
        if predecessor not in subgraph
    }
    augmented = set(subgraph) | inputs
    if not augmented:
        return True

    start = next(iter(augmented))
    visited = {start}
    stack = [start]
    while stack:
        node = stack.pop()
        neighbors: Iterator[NodeT]
        if node in graph:
            neighbors = iter(
                tuple(graph.predecessors(node)) + tuple(graph.successors(node))
            )
        else:
            neighbors = iter(())
        for neighbor in neighbors:
            if neighbor not in augmented or neighbor in visited:
                continue
            if node in inputs and neighbor in inputs:
                continue
            visited.add(neighbor)
            stack.append(neighbor)
    return len(visited) == len(augmented)


def _passes_alternate_connectivity(
    alternate_graph: nx.DiGraph[NodeT] | None,
    subgraph: set[NodeT],
    *,
    alternate_connected_only: bool,
) -> bool:
    if not alternate_connected_only:
        return True
    assert alternate_graph is not None
    return _is_connected_with_inputs(alternate_graph, subgraph)


def _closure_in_graph(
    graph: nx.DiGraph[NodeT],
    subgraph: set[NodeT],
) -> set[NodeT]:
    predecessors: set[NodeT] = set()
    successors: set[NodeT] = set()
    for node in subgraph:
        if node not in graph:
            continue
        predecessors.update(nx.ancestors(graph, node))
        successors.update(nx.descendants(graph, node))
    return set(subgraph) | (predecessors & successors)


def _dual_closure(
    graph: nx.DiGraph[NodeT],
    alternate_graph: nx.DiGraph[NodeT] | None,
    subgraph: set[NodeT],
) -> set[NodeT]:
    closed = set(subgraph)
    while True:
        next_closed = _closure_in_graph(graph, closed)
        if alternate_graph is not None:
            next_closed |= _closure_in_graph(alternate_graph, closed)
        if next_closed == closed:
            return closed
        closed = next_closed


def _is_valid_convex_candidate(
    graph: nx.DiGraph[NodeT],
    alternate_graph: nx.DiGraph[NodeT] | None,
    payload: GraphInput,
    node_order: tuple[NodeT, ...],
    subgraph: set[NodeT],
    *,
    max_num_inputs: int,
    max_num_outputs: int,
    max_subgraph_size: int,
    require_positive_outputs: bool,
    alternate_connected_only: bool,
) -> bool:
    if any(node not in graph for node in subgraph):
        return False
    if max_subgraph_size >= 0 and len(subgraph) > max_subgraph_size:
        return False

    body_forbidden = {
        node_order[index]
        for index, forbidden in enumerate(payload.body_forbidden)
        if forbidden
    }
    if subgraph & body_forbidden:
        return False

    inputs = {
        predecessor
        for node in subgraph
        for predecessor in graph.predecessors(node)
        if predecessor not in subgraph
    }
    if len(inputs) > max_num_inputs:
        return False
    input_forbidden = {
        node_order[index]
        for index, forbidden in enumerate(payload.input_forbidden)
        if forbidden
    }
    if inputs & input_forbidden:
        return False

    num_outputs = _num_outputs(graph, subgraph)
    if num_outputs > max_num_outputs:
        return False
    if require_positive_outputs and num_outputs == 0:
        return False

    if _dual_closure(graph, alternate_graph, subgraph) != subgraph:
        return False
    return _passes_alternate_connectivity(
        alternate_graph,
        subgraph,
        alternate_connected_only=alternate_connected_only,
    )


def _iter_successor_expansions(
    graph: nx.DiGraph[NodeT],
    alternate_graph: nx.DiGraph[NodeT] | None,
    payload: GraphInput,
    node_order: tuple[NodeT, ...],
    subgraph: set[NodeT],
    *,
    max_num_inputs: int,
    max_num_outputs: int,
    max_subgraph_size: int,
    require_positive_outputs: bool,
    alternate_connected_only: bool,
) -> Iterator[set[NodeT]]:
    seen = {frozenset(subgraph)}
    stack = [set(subgraph)]
    while stack:
        current = stack.pop()
        candidates = {
            successor
            for node in current
            if node in graph
            for successor in graph.successors(node)
            if successor not in current
        }
        for candidate in candidates:
            expanded = _dual_closure(
                graph,
                alternate_graph,
                current | {candidate},
            )
            key = frozenset(expanded)
            if key in seen:
                continue
            seen.add(key)
            if not _is_valid_convex_candidate(
                graph,
                alternate_graph,
                payload,
                node_order,
                expanded,
                max_num_inputs=max_num_inputs,
                max_num_outputs=max_num_outputs,
                max_subgraph_size=max_subgraph_size,
                require_positive_outputs=require_positive_outputs,
                alternate_connected_only=alternate_connected_only,
            ):
                continue
            yield expanded
            stack.append(expanded)


def graph_to_input(
    graph: nx.DiGraph[NodeT],
    *,
    alternate_graph: nx.DiGraph[NodeT] | None = None,
    weighted: bool = False,
    weight_attr: str = "weight",
    forbidden_attr: str | None = "forbidden",
    body_forbidden_attr: str | None = None,
    input_forbidden_attr: str | None = None,
    forbid_sources_and_sinks: bool = True,
    ordering: Ordering = "toposort",
    name: str | None = None,
) -> tuple[GraphInput, tuple[NodeT, ...]]:  # second tuple item is the reverse mapping for the ints
    if not nx.is_directed_acyclic_graph(graph):
        raise ValueError("Graph must be a DAG")
    graph_nodes = set(graph.nodes)
    alternate_nodes: set[NodeT] = set()
    graph_shared_forbidden = (
        {
            node
            for node in graph_nodes
            if _as_bool(graph.nodes[node].get(forbidden_attr, False))
        }
        if forbidden_attr is not None
        else set()
    )
    graph_body_forbidden = set(graph_shared_forbidden)
    graph_input_forbidden = set(graph_shared_forbidden)
    if body_forbidden_attr is not None:
        graph_body_forbidden |= {
            node
            for node in graph_nodes
            if _as_bool(graph.nodes[node].get(body_forbidden_attr, False))
        }
    if input_forbidden_attr is not None:
        graph_input_forbidden |= {
            node
            for node in graph_nodes
            if _as_bool(graph.nodes[node].get(input_forbidden_attr, False))
        }
    alternate_shared_forbidden: set[NodeT] = set()
    alternate_body_forbidden: set[NodeT] = set()
    alternate_input_forbidden: set[NodeT] = set()
    if alternate_graph is not None:
        if not nx.is_directed_acyclic_graph(alternate_graph):
            raise ValueError("alternate_graph must be a DAG")
        alternate_nodes = set(alternate_graph.nodes)
        alternate_shared_forbidden = (
            {
                node
                for node in alternate_nodes
                if _as_bool(alternate_graph.nodes[node].get(forbidden_attr, False))
            }
            if forbidden_attr is not None
            else set()
        )
        alternate_body_forbidden = set(alternate_shared_forbidden)
        alternate_input_forbidden = set(alternate_shared_forbidden)
        if body_forbidden_attr is not None:
            alternate_body_forbidden |= {
                node
                for node in alternate_nodes
                if _as_bool(alternate_graph.nodes[node].get(body_forbidden_attr, False))
            }
        if input_forbidden_attr is not None:
            alternate_input_forbidden |= {
                node
                for node in alternate_nodes
                if _as_bool(alternate_graph.nodes[node].get(input_forbidden_attr, False))
            }
        missing_from_alternate = graph_nodes - alternate_nodes
        if missing_from_alternate - (graph_body_forbidden | graph_input_forbidden):
            raise ValueError(
                "alternate_graph may only omit nodes that are forbidden in graph"
            )
        missing_from_graph = alternate_nodes - graph_nodes
        if missing_from_graph - (alternate_body_forbidden | alternate_input_forbidden):
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
    body_forbidden_nodes = graph_body_forbidden | alternate_body_forbidden
    input_forbidden_nodes = graph_input_forbidden | alternate_input_forbidden
    forbidden_nodes = body_forbidden_nodes & input_forbidden_nodes

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
    payload.body_forbidden = [
        1 if node in body_forbidden_nodes else 0
        for node in node_order
    ]
    payload.input_forbidden = [
        1 if node in input_forbidden_nodes else 0
        for node in node_order
    ]
    return payload, tuple(node_order)


def _iter_exact_then_sampled(
    exact_factory: Callable[[int | None], Iterator[set[NodeT]]],
    sampled_factory: Callable[[], Iterator[set[NodeT]]],
    *,
    threshold: int,
    work_limit: int = 0,
) -> Iterator[set[NodeT]]:
    if threshold <= 0:
        yield from sampled_factory()
        return

    buffered: list[set[NodeT]] = []
    exact_iter = exact_factory(work_limit if work_limit > 0 else None)
    use_sampled = False
    try:
        for subgraph in exact_iter:
            buffered.append(subgraph)
            if len(buffered) >= threshold:
                buffered.clear()
                use_sampled = True
                break
    except _ExhaustiveWorkLimitReached:
        buffered.clear()
        use_sampled = True
    finally:
        close = getattr(exact_iter, "close", None)
        if close is not None:
            close()

    if use_sampled:
        yield from sampled_factory()
    else:
        yield from buffered


def _iter_convex_subgraphs(
    graph: nx.DiGraph[NodeT],
    alternate_graph: nx.DiGraph[NodeT] | None,
    payload: GraphInput,
    node_order: tuple[NodeT, ...],
    *,
    max_num_inputs: int,
    max_num_outputs: int,
    max_subgraph_size: int,
    weighted: bool,
    weight_attr: str,
    forbidden_attr: str | None,
    forbid_sources_and_sinks: bool,
    allow_zero_outputs: bool,
    connected_only: bool,
    alternate_connected_only: bool,
    ordering: Ordering,
    max_queue_size: int,
    max_work: int = 0,
) -> Iterator[set[NodeT]]:
    seen: set[frozenset[NodeT]] = set()
    require_positive_outputs = not allow_zero_outputs and max_num_outputs > 0

    def emit_with_expansions(subgraph_nodes: set[NodeT]) -> Iterator[set[NodeT]]:
        key = frozenset(subgraph_nodes)
        if key in seen:
            return
        seen.add(key)
        yield subgraph_nodes
        if not alternate_connected_only or connected_only or max_work > 0:
            return
        for expanded in _iter_successor_expansions(
            graph,
            alternate_graph,
            payload,
            node_order,
            subgraph_nodes,
            max_num_inputs=max_num_inputs,
            max_num_outputs=max_num_outputs,
            max_subgraph_size=max_subgraph_size,
            require_positive_outputs=require_positive_outputs,
            alternate_connected_only=alternate_connected_only,
        ):
            expanded_key = frozenset(expanded)
            if expanded_key in seen:
                continue
            seen.add(expanded_key)
            yield expanded

    iterator = iter_all_graph_input(
        payload,
        max_num_inputs,
        max_num_outputs,
        max_subgraph_size,
        max_queue_size=max_queue_size,
        connected_only=connected_only,
        max_work=max_work,
    )
    try:
        for subgraph in iterator:
            subgraph_nodes = {node_order[index] for index in subgraph}
            if require_positive_outputs and _num_outputs(graph, subgraph_nodes) == 0:
                continue
            if not _passes_alternate_connectivity(
                alternate_graph,
                subgraph_nodes,
                alternate_connected_only=alternate_connected_only,
            ):
                continue
            yield from emit_with_expansions(subgraph_nodes)
    finally:
        iterator.close()
    if iterator.hit_work_limit():
        raise _ExhaustiveWorkLimitReached

    if forbid_sources_and_sinks:
        for index, node in enumerate(node_order):
            if node not in graph:
                continue
            if graph.in_degree(node) > 0 and graph.out_degree(node) > 0:
                continue
            key = frozenset({node})
            if key in seen:
                continue
            if max_subgraph_size >= 0 and max_subgraph_size < 1:
                continue
            if payload.body_forbidden[index]:
                continue
            subgraph_nodes = {node}
            num_outputs = _num_outputs(graph, subgraph_nodes)
            if num_outputs > max_num_outputs:
                continue
            if require_positive_outputs and num_outputs == 0:
                continue
            num_inputs = sum(1 for predecessor in graph.predecessors(node))
            if num_inputs > max_num_inputs:
                continue
            if not _passes_alternate_connectivity(
                alternate_graph,
                subgraph_nodes,
                alternate_connected_only=alternate_connected_only,
            ):
                continue
            yield from emit_with_expansions(subgraph_nodes)

    if allow_zero_outputs:
        iterator = iter_all_graph_input(
            payload,
            max_num_inputs,
            0,
            max_subgraph_size,
            max_queue_size=max_queue_size,
            connected_only=connected_only,
            max_work=max_work,
        )
        try:
            for subgraph in iterator:
                subgraph_nodes = {node_order[index] for index in subgraph}
                if not _passes_alternate_connectivity(
                    alternate_graph,
                    subgraph_nodes,
                    alternate_connected_only=alternate_connected_only,
                ):
                    continue
                yield from emit_with_expansions(subgraph_nodes)
        finally:
            iterator.close()
        if iterator.hit_work_limit():
            raise _ExhaustiveWorkLimitReached


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
