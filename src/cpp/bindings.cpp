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
    std::vector<double> weights;
    std::vector<uint8_t> forbidden;
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
    for (const auto &[source, target] : input.edges) {
        if (source < 0 || source >= input.num_nodes || target < 0 || target >= input.num_nodes) {
            throw nb::value_error("edge endpoint out of range");
        }
    }
}

std::unique_ptr<DFG> make_dfg(const GraphInput &input)
{
    validate_graph_input(input);
    auto dfg = std::make_unique<DFG>(
        input.name,
        input.num_nodes,
        input.frequency,
        input.forbid_sources_and_sinks);
    for (int node = 0; node < input.num_nodes; ++node) {
        if (!input.weights.empty())
            dfg->weight(node) = input.weights[static_cast<std::size_t>(node)];
        if (!input.forbidden.empty() &&
            input.forbidden[static_cast<std::size_t>(node)] != 0) {
            dfg->set_forbidden(node);
        }
    }
    for (const auto &[source, target] : input.edges)
        dfg->add_edge(source, target);
    dfg->index();
    return dfg;
}

class ExhaustiveSubgraphIterator {
public:
    ExhaustiveSubgraphIterator(const GraphInput &input,
                               int max_num_inputs,
                               int max_num_outputs,
                               std::size_t max_queue_size)
        : dfg_(make_dfg(input))
        , max_num_inputs_(max_num_inputs)
        , max_num_outputs_(max_num_outputs)
        , max_queue_size_(max_queue_size)
    {
        if (max_num_inputs_ < 0 || max_num_outputs_ < 0)
            throw nb::value_error("I/O limits must be non-negative");
        if (max_queue_size_ == 0)
            throw nb::value_error("max_queue_size must be positive");
        if (dfg_->forbidden().size() == dfg_->num_nodes()) {
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
                *dfg_,
                max_num_inputs_,
                max_num_outputs_,
                [this](const IOSubgraph &subgraph) {
                    push_result(to_vector(subgraph.nodes()));
                });
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

    std::unique_ptr<DFG> dfg_;
    int max_num_inputs_;
    int max_num_outputs_;
    std::size_t max_queue_size_;
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
                              const std::string &iteration_type,
                              int flags)
{
    if (max_num_inputs < 0 || max_num_outputs < 0)
        throw nb::value_error("I/O limits must be non-negative");

    auto dfg = make_dfg(input);
    if (dfg->forbidden().size() == dfg->num_nodes())
        return {};

    StderrSilencer silence;
    MVSFinder finder(dfg.get());
    const auto output = finder.enumerate(
        max_num_inputs,
        max_num_outputs,
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
                                  int max_num_outputs)
{
    if (max_num_inputs < 0 || max_num_outputs < 0)
        throw nb::value_error("I/O limits must be non-negative");

    auto dfg = make_dfg(input);
    if (dfg->forbidden().size() == dfg->num_nodes())
        return {};

    SolveResult result;
    StderrSilencer silence;
    vs_enumerate(
        *dfg,
        max_num_inputs,
        max_num_outputs,
        [&result](const IOSubgraph &subgraph) {
            result.max_weight = std::max(result.max_weight, subgraph.weight());
            result.subgraphs.push_back(to_vector(subgraph.nodes()));
        });
    return result;
}

std::shared_ptr<ExhaustiveSubgraphIterator> iter_all_graph_input(
    const GraphInput &input,
    int max_num_inputs,
    int max_num_outputs,
    std::size_t max_queue_size)
{
    return std::make_shared<ExhaustiveSubgraphIterator>(
        input, max_num_inputs, max_num_outputs, max_queue_size);
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
        .def_rw("weights", &GraphInput::weights)
        .def_rw("forbidden", &GraphInput::forbidden)
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
        nb::arg("iteration_type") = "linear-rev",
        nb::arg("flags") = 0xff);
    m.def(
        "solve_all_graph_input",
        &solve_all_graph_input,
        nb::arg("graph_input"),
        nb::arg("max_num_inputs"),
        nb::arg("max_num_outputs"));
    m.def(
        "iter_all_graph_input",
        &iter_all_graph_input,
        nb::arg("graph_input"),
        nb::arg("max_num_inputs"),
        nb::arg("max_num_outputs"),
        nb::arg("max_queue_size") = 128);
}
