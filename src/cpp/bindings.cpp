#include <nanobind/nanobind.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include "dfg.h"
#include "mvs.h"

#include <cstdint>
#include <string>
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

SolveResult solve_graph_input(const GraphInput &input,
                              int max_num_inputs,
                              int max_num_outputs,
                              const std::string &iteration_type,
                              int flags)
{
    validate_graph_input(input);
    if (max_num_inputs < 0 || max_num_outputs < 0)
        throw nb::value_error("I/O limits must be non-negative");

    DFG dfg(input.name, input.num_nodes, input.frequency);
    for (int node = 0; node < input.num_nodes; ++node) {
        if (!input.weights.empty())
            dfg.weight(node) = input.weights[static_cast<std::size_t>(node)];
        if (!input.forbidden.empty() &&
            input.forbidden[static_cast<std::size_t>(node)] != 0) {
            dfg.set_forbidden(node);
        }
    }
    for (const auto &[source, target] : input.edges)
        dfg.add_edge(source, target);
    dfg.index();

    if (dfg.forbidden().size() == dfg.num_nodes())
        return {};

    StderrSilencer silence;
    MVSFinder finder(&dfg);
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
        .def_rw("frequency", &GraphInput::frequency);

    nb::class_<SolveResult>(m, "SolveResult")
        .def(nb::init<>())
        .def_rw("max_weight", &SolveResult::max_weight)
        .def_rw("subgraphs", &SolveResult::subgraphs);

    m.def(
        "solve_graph_input",
        &solve_graph_input,
        nb::arg("graph_input"),
        nb::arg("max_num_inputs"),
        nb::arg("max_num_outputs"),
        nb::arg("iteration_type") = "linear-rev",
        nb::arg("flags") = 0xff);
}
