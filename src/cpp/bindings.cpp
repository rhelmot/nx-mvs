#include <nanobind/nanobind.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/shared_ptr.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include "dfg.h"
#include "mvs.h"
#include "vs.h"

#include <algorithm>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <exception>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace nb = nanobind;

struct GraphInput {
    std::string name;
    int num_nodes = 0;
    std::vector<std::pair<int, int>> edges;
    std::vector<std::pair<int, int>> alternate_edges;
    std::vector<double> weights;
    std::vector<uint8_t> forbidden;
    std::vector<uint8_t> body_forbidden;
    std::vector<uint8_t> input_forbidden;
    int frequency = 0;
    bool forbid_sources_and_sinks = true;
};

struct SolveResult {
    double max_weight = 0.0;
    std::vector<std::vector<int>> subgraphs;
};

namespace {

class NullBuffer : public std::streambuf {
public:
    int overflow(int c) override { return c; }
};

class StderrSilencer {
public:
    StderrSilencer()
        : previous_(std::cerr.rdbuf(&null_buffer_))
    {
    }

    ~StderrSilencer() { std::cerr.rdbuf(previous_); }

private:
    NullBuffer null_buffer_;
    std::streambuf *previous_;
};

class EnumerationStopped : public std::exception {
};

MVSFinder::IterType parse_iteration_type(const std::string &iteration_type)
{
    if (iteration_type == "linear")
        return MVSFinder::IterType::LINEAR;
    if (iteration_type == "linear-rev")
        return MVSFinder::IterType::LINEAR_REV;
    if (iteration_type == "binary-search")
        return MVSFinder::IterType::BINARY_SEARCH;
    throw nb::value_error("iteration_type must be 'linear', 'linear-rev', or 'binary-search'");
}

std::vector<int> to_vector(const intset &set)
{
    std::vector<int> out;
    out.reserve(set.size());
    for (const auto &value : set)
        out.push_back(value);
    return out;
}

void validate_graph_input(const GraphInput &input)
{
    if (input.num_nodes < 0)
        throw nb::value_error("num_nodes must be non-negative");
    if (!input.weights.empty() && input.weights.size() != input.num_nodes)
        throw nb::value_error("weights must be empty or have length num_nodes");
    if (!input.forbidden.empty() && input.forbidden.size() != input.num_nodes)
        throw nb::value_error("forbidden must be empty or have length num_nodes");
    if (!input.body_forbidden.empty() &&
        input.body_forbidden.size() != input.num_nodes)
        throw nb::value_error("body_forbidden must be empty or have length num_nodes");
    if (!input.input_forbidden.empty() &&
        input.input_forbidden.size() != input.num_nodes)
        throw nb::value_error("input_forbidden must be empty or have length num_nodes");
    for (const auto &[source, target] : input.edges) {
        if (source < 0 || source >= input.num_nodes || target < 0 || target >= input.num_nodes) {
            throw nb::value_error("edge endpoint out of range");
        }
    }
    for (const auto &[source, target] : input.alternate_edges) {
        if (source < 0 || source >= input.num_nodes || target < 0 || target >= input.num_nodes) {
            throw nb::value_error("alternate edge endpoint out of range");
        }
    }
}

std::pair<bool, bool> node_forbidden_flags(const GraphInput &input, int node)
{
    bool body_forbidden = false;
    bool input_forbidden = false;
    if (!input.forbidden.empty() &&
        input.forbidden[static_cast<std::size_t>(node)] != 0) {
        body_forbidden = true;
        input_forbidden = true;
    }
    if (!input.body_forbidden.empty() &&
        input.body_forbidden[static_cast<std::size_t>(node)] != 0)
        body_forbidden = true;
    if (!input.input_forbidden.empty() &&
        input.input_forbidden[static_cast<std::size_t>(node)] != 0)
        input_forbidden = true;
    return {body_forbidden, input_forbidden};
}

std::unique_ptr<DFG> make_graph(const GraphInput &input,
                                bool materialize_opt_in_terminals = false)
{
    validate_graph_input(input);
    if (materialize_opt_in_terminals && !input.forbid_sources_and_sinks) {
        std::vector<uint8_t> has_in(static_cast<std::size_t>(input.num_nodes), 0);
        std::vector<uint8_t> has_out(static_cast<std::size_t>(input.num_nodes), 0);
        for (const auto &[source, target] : input.edges) {
            has_out[static_cast<std::size_t>(source)] = 1;
            has_in[static_cast<std::size_t>(target)] = 1;
        }

        int extra_nodes = 0;
        for (int node = 0; node < input.num_nodes; ++node) {
            if (has_in[static_cast<std::size_t>(node)] == 0)
                ++extra_nodes;
            if (has_out[static_cast<std::size_t>(node)] == 0)
                ++extra_nodes;
        }

        auto graph = std::make_unique<DFG>(
            input.name,
            input.num_nodes + extra_nodes,
            input.frequency,
            true);
        for (int node = 0; node < input.num_nodes; ++node) {
            if (!input.weights.empty())
                graph->weight(node) = input.weights[static_cast<std::size_t>(node)];
            auto [body_forbidden, input_forbidden] =
                node_forbidden_flags(input, node);
            if (body_forbidden)
                graph->set_body_forbidden(node);
            if (input_forbidden)
                graph->set_input_forbidden(node);
        }
        for (const auto &[source, target] : input.edges)
            graph->add_edge(source, target);

        int next_node = input.num_nodes;
        for (int node = 0; node < input.num_nodes; ++node) {
            if (has_in[static_cast<std::size_t>(node)] == 0) {
                graph->weight(next_node) = 0;
                graph->set_forbidden(next_node);
                graph->add_edge(next_node, node);
                ++next_node;
            }
            if (has_out[static_cast<std::size_t>(node)] == 0) {
                graph->weight(next_node) = 0;
                graph->set_forbidden(next_node);
                graph->add_edge(node, next_node);
                ++next_node;
            }
        }
        graph->index();
        return graph;
    }

    auto graph = std::make_unique<DFG>(
        input.name,
        input.num_nodes,
        input.frequency,
        input.forbid_sources_and_sinks);
    for (int node = 0; node < input.num_nodes; ++node) {
        if (!input.weights.empty())
            graph->weight(node) = input.weights[static_cast<std::size_t>(node)];
        auto [body_forbidden, input_forbidden] = node_forbidden_flags(input, node);
        if (body_forbidden)
            graph->set_body_forbidden(node);
        if (input_forbidden)
            graph->set_input_forbidden(node);
    }
    for (const auto &[source, target] : input.edges)
        graph->add_edge(source, target);
    graph->index();
    return graph;
}

std::unique_ptr<DFG> make_alternate_graph(const GraphInput &input)
{
    if (input.alternate_edges.empty())
        return nullptr;

    auto graph = std::make_unique<DFG>(
        input.name + "_alternate",
        input.num_nodes,
        input.frequency,
        false);
    for (int node = 0; node < input.num_nodes; ++node) {
        auto [body_forbidden, input_forbidden] = node_forbidden_flags(input, node);
        if (body_forbidden)
            graph->set_body_forbidden(node);
        if (input_forbidden)
            graph->set_input_forbidden(node);
    }
    for (const auto &[source, target] : input.alternate_edges)
        graph->add_edge(source, target);
    graph->index();
    return graph;
}

class ExhaustiveSubgraphIterator {
public:
    ExhaustiveSubgraphIterator(const GraphInput &input,
                               int max_num_inputs,
                               int max_num_outputs,
                               int max_subgraph_size,
                               std::size_t max_queue_size,
                               bool connected_only)
        : graph_(make_graph(input))
        , alternate_graph_(make_alternate_graph(input))
        , max_num_inputs_(max_num_inputs)
        , max_num_outputs_(max_num_outputs)
        , max_subgraph_size_(max_subgraph_size)
        , max_queue_size_(max_queue_size)
        , connected_only_(connected_only)
    {
        if (max_num_inputs_ < 0 || max_num_outputs_ < 0)
            throw nb::value_error("I/O limits must be non-negative");
        if (max_subgraph_size_ < -1)
            throw nb::value_error("max_subgraph_size must be -1 or non-negative");
        if (max_queue_size_ == 0)
            throw nb::value_error("max_queue_size must be positive");
        if (graph_->forbidden().size() == graph_->num_nodes()) {
            done_ = true;
            return;
        }

        worker_ = std::thread(&ExhaustiveSubgraphIterator::run, this);
    }

    ~ExhaustiveSubgraphIterator() { close(); }

    ExhaustiveSubgraphIterator(const ExhaustiveSubgraphIterator &) = delete;
    ExhaustiveSubgraphIterator &operator=(const ExhaustiveSubgraphIterator &) = delete;

    ExhaustiveSubgraphIterator &iter() { return *this; }

    std::vector<int> next()
    {
        std::unique_lock<std::mutex> lock(mutex_);
        while (queue_.empty() && !done_) {
            {
                nb::gil_scoped_release release;
                cv_.wait(lock);
            }
        }

        if (!queue_.empty()) {
            std::vector<int> next_subgraph = std::move(queue_.front());
            queue_.pop_front();
            cv_.notify_all();
            return next_subgraph;
        }

        if (worker_exception_ != nullptr)
            std::rethrow_exception(worker_exception_);
        throw nb::stop_iteration();
    }

    void close()
    {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            stop_requested_ = true;
            cv_.notify_all();
        }
        if (worker_.joinable())
            worker_.join();
    }

private:
    void push_result(std::vector<int> subgraph)
    {
        std::unique_lock<std::mutex> lock(mutex_);
        while (queue_.size() >= max_queue_size_ && !stop_requested_) {
            cv_.wait(lock);
        }
        if (stop_requested_)
            throw EnumerationStopped();

        queue_.push_back(std::move(subgraph));
        cv_.notify_all();
    }

    void run()
    {
        try {
            StderrSilencer silence;
            vs_enumerate(
                *graph_,
                max_num_inputs_,
                max_num_outputs_,
                max_subgraph_size_,
                alternate_graph_.get(),
                [this](const IOSubgraph &subgraph) {
                    push_result(to_vector(subgraph.nodes()));
                },
                connected_only_);
        } catch (const EnumerationStopped &) {
        } catch (...) {
            std::lock_guard<std::mutex> lock(mutex_);
            worker_exception_ = std::current_exception();
        }

        {
            std::lock_guard<std::mutex> lock(mutex_);
            done_ = true;
        }
        cv_.notify_all();
    }

    std::unique_ptr<DFG> graph_;
    std::unique_ptr<DFG> alternate_graph_;
    int max_num_inputs_;
    int max_num_outputs_;
    int max_subgraph_size_;
    std::size_t max_queue_size_;
    bool connected_only_;
    std::deque<std::vector<int>> queue_;
    std::mutex mutex_;
    std::condition_variable cv_;
    std::thread worker_;
    std::exception_ptr worker_exception_;
    bool done_ = false;
    bool stop_requested_ = false;
};

SolveResult solve_graph_input(const GraphInput &input,
                              int max_num_inputs,
                              int max_num_outputs,
                              int max_subgraph_size,
                              const std::string &iteration_type,
                              int flags)
{
    if (max_num_inputs < 0 || max_num_outputs < 0)
        throw nb::value_error("I/O limits must be non-negative");
    if (max_subgraph_size < -1)
        throw nb::value_error("max_subgraph_size must be -1 or non-negative");
    if (!input.alternate_edges.empty())
        throw nb::value_error(
            "alternate_edges are only supported by exhaustive enumeration paths");

    auto graph = make_graph(input, true);
    if (graph->forbidden().size() == graph->num_nodes())
        return {};

    StderrSilencer silence;
    MVSFinder finder(graph.get());
    const auto output = finder.enumerate(
        max_num_inputs,
        max_num_outputs,
        max_subgraph_size,
        parse_iteration_type(iteration_type),
        static_cast<uint8_t>(flags));

    SolveResult result;
    if (!output.empty())
        result.max_weight = output.front().weight();
    result.subgraphs.reserve(output.size());
    for (const auto &subgraph : output)
        result.subgraphs.push_back(to_vector(subgraph.nodes()));
    return result;
}

SolveResult solve_all_graph_input(const GraphInput &input,
                                  int max_num_inputs,
                                  int max_num_outputs,
                                  int max_subgraph_size,
                                  bool connected_only)
{
    if (max_num_inputs < 0 || max_num_outputs < 0)
        throw nb::value_error("I/O limits must be non-negative");
    if (max_subgraph_size < -1)
        throw nb::value_error("max_subgraph_size must be -1 or non-negative");

    auto graph = make_graph(input);
    auto alternate_graph = make_alternate_graph(input);
    if (graph->forbidden().size() == graph->num_nodes())
        return {};

    SolveResult result;
    StderrSilencer silence;
    vs_enumerate(
        *graph,
        max_num_inputs,
        max_num_outputs,
        max_subgraph_size,
        alternate_graph.get(),
        [&result](const IOSubgraph &subgraph) {
            result.max_weight = std::max(result.max_weight, subgraph.weight());
            result.subgraphs.push_back(to_vector(subgraph.nodes()));
        },
        connected_only);
    return result;
}

SolveResult sample_zero_output_graph_input(const GraphInput &input,
                                           int max_num_inputs,
                                           int max_subgraph_size,
                                           int max_states_expanded,
                                           int max_samples,
                                           int max_children_per_state,
                                           int size_bin_width,
                                           int thicken_radius,
                                           bool bucket_by_num_inputs,
                                           int minimal_node_bin_width)
{
    if (max_num_inputs < 0)
        throw nb::value_error("I/O limits must be non-negative");
    if (max_subgraph_size < -1)
        throw nb::value_error("max_subgraph_size must be -1 or non-negative");
    if (max_states_expanded < 0 || max_samples < 0)
        throw nb::value_error("sampling budgets must be non-negative");
    if (max_children_per_state <= 0)
        throw nb::value_error("max_children_per_state must be positive");
    if (size_bin_width <= 0)
        throw nb::value_error("size_bin_width must be positive");
    if (thicken_radius < 0)
        throw nb::value_error("thicken_radius must be non-negative");
    if (minimal_node_bin_width < 0)
        throw nb::value_error("minimal_node_bin_width must be non-negative");

    auto graph = make_graph(input);
    auto alternate_graph = make_alternate_graph(input);
    if (graph->forbidden().size() == graph->num_nodes())
        return {};

    SolveResult result;
    StderrSilencer silence;
    vs_sample_zero_output_connected(
        *graph,
        max_num_inputs,
        max_subgraph_size,
        alternate_graph.get(),
        [&result](const IOSubgraph &subgraph) {
            result.max_weight = std::max(result.max_weight, subgraph.weight());
            result.subgraphs.push_back(to_vector(subgraph.nodes()));
        },
        max_states_expanded,
        max_samples,
        max_children_per_state,
        size_bin_width,
        thicken_radius,
        bucket_by_num_inputs,
        minimal_node_bin_width);
    return result;
}

SolveResult grow_zero_output_graph_input(const GraphInput &input,
                                         std::vector<int> seed_nodes,
                                         int max_num_inputs,
                                         int max_subgraph_size,
                                         nb::object oracle,
                                         nb::object initial_oracle_state)
{
    if (max_num_inputs < 0)
        throw nb::value_error("I/O limits must be non-negative");
    if (max_subgraph_size < -1)
        throw nb::value_error("max_subgraph_size must be -1 or non-negative");

    auto graph = make_graph(input);
    auto alternate_graph = make_alternate_graph(input);
    if (graph->forbidden().size() == graph->num_nodes())
        return {};

    intset seed(static_cast<unsigned>(graph->num_nodes()));
    for (const auto &node : seed_nodes) {
        if (node < 0 || node >= graph->num_nodes())
            throw nb::value_error("seed node index out of range");
        seed.add(static_cast<unsigned>(node));
    }

    SolveResult result;
    std::vector<nb::object> oracle_states;
    oracle_states.push_back(initial_oracle_state.is_valid() ? initial_oracle_state
                                                            : nb::none());
    StderrSilencer silence;
    nb::gil_scoped_release release;
    vs_grow_zero_output_connected(
        *graph,
        seed,
        max_num_inputs,
        max_subgraph_size,
        alternate_graph.get(),
        0,
        [&result, &oracle, &oracle_states](const IOSubgraph &subgraph,
                                           std::size_t state_token)
            -> std::optional<std::size_t> {
            auto nodes = to_vector(subgraph.nodes());
            result.max_weight = std::max(result.max_weight, subgraph.weight());
            result.subgraphs.push_back(nodes);
            if (oracle.is_none())
                return state_token;

            nb::gil_scoped_acquire acquire;
            nb::object outcome = oracle(oracle_states.at(state_token), nodes);
            if (outcome.is_none())
                return std::nullopt;
            if (nb::isinstance<nb::bool_>(outcome)) {
                return nb::cast<bool>(outcome) ? std::optional<std::size_t>(state_token)
                                               : std::nullopt;
            }
            oracle_states.push_back(std::move(outcome));
            return oracle_states.size() - 1;
        });
    return result;
}

std::shared_ptr<ExhaustiveSubgraphIterator> iter_all_graph_input(
    const GraphInput &input,
    int max_num_inputs,
    int max_num_outputs,
    int max_subgraph_size,
    std::size_t max_queue_size,
    bool connected_only)
{
    return std::make_shared<ExhaustiveSubgraphIterator>(
        input,
        max_num_inputs,
        max_num_outputs,
        max_subgraph_size,
        max_queue_size,
        connected_only);
}

}

NB_MODULE(_native, m)
{
    m.doc() = "nanobind adapter for the mvs solver";

    nb::class_<GraphInput>(m, "GraphInput")
        .def(nb::init<>())
        .def_rw("name", &GraphInput::name)
        .def_rw("num_nodes", &GraphInput::num_nodes)
        .def_rw("edges", &GraphInput::edges)
        .def_rw("alternate_edges", &GraphInput::alternate_edges)
        .def_rw("weights", &GraphInput::weights)
        .def_rw("forbidden", &GraphInput::forbidden)
        .def_rw("body_forbidden", &GraphInput::body_forbidden)
        .def_rw("input_forbidden", &GraphInput::input_forbidden)
        .def_rw("frequency", &GraphInput::frequency)
        .def_rw("forbid_sources_and_sinks", &GraphInput::forbid_sources_and_sinks);

    nb::class_<SolveResult>(m, "SolveResult")
        .def(nb::init<>())
        .def_rw("max_weight", &SolveResult::max_weight)
        .def_rw("subgraphs", &SolveResult::subgraphs);

    nb::class_<ExhaustiveSubgraphIterator>(m, "ExhaustiveSubgraphIterator")
        .def("__iter__", &ExhaustiveSubgraphIterator::iter, nb::rv_policy::reference_internal)
        .def("__next__", &ExhaustiveSubgraphIterator::next)
        .def("close", &ExhaustiveSubgraphIterator::close);

    m.def(
        "solve_graph_input",
        &solve_graph_input,
        nb::arg("graph_input"),
        nb::arg("max_num_inputs"),
        nb::arg("max_num_outputs"),
        nb::arg("max_subgraph_size") = -1,
        nb::arg("iteration_type") = "linear-rev",
        nb::arg("flags") = 0xff);
    m.def(
        "solve_all_graph_input",
        &solve_all_graph_input,
        nb::arg("graph_input"),
        nb::arg("max_num_inputs"),
        nb::arg("max_num_outputs"),
        nb::arg("max_subgraph_size") = -1,
        nb::arg("connected_only") = false);
    m.def(
        "iter_all_graph_input",
        &iter_all_graph_input,
        nb::arg("graph_input"),
        nb::arg("max_num_inputs"),
        nb::arg("max_num_outputs"),
        nb::arg("max_subgraph_size") = -1,
        nb::arg("max_queue_size") = 128,
        nb::arg("connected_only") = false);
    m.def(
        "sample_zero_output_graph_input",
        &sample_zero_output_graph_input,
        nb::arg("graph_input"),
        nb::arg("max_num_inputs"),
        nb::arg("max_subgraph_size") = -1,
        nb::arg("max_states_expanded") = 10000,
        nb::arg("max_samples") = 1000,
        nb::arg("max_children_per_state") = 2,
        nb::arg("size_bin_width") = 4,
        nb::arg("thicken_radius") = 1,
        nb::arg("bucket_by_num_inputs") = true,
        nb::arg("minimal_node_bin_width") = 1);
    m.def(
        "grow_zero_output_graph_input",
        &grow_zero_output_graph_input,
        nb::arg("graph_input"),
        nb::arg("seed_nodes"),
        nb::arg("max_num_inputs"),
        nb::arg("max_subgraph_size") = -1,
        nb::arg("oracle") = nb::none(),
        nb::arg("initial_oracle_state") = nb::none());
}
