cmake_minimum_required(VERSION 3.18)
project(mvs LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

find_package(Python REQUIRED COMPONENTS Interpreter Development.Module)

execute_process(
    COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
    OUTPUT_VARIABLE nanobind_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
)
find_package(nanobind CONFIG REQUIRED)

find_package(nlohmann_json CONFIG QUIET)

if(NOT nlohmann_json_FOUND)
    include(FetchContent)
    FetchContent_Declare(
        json
        URL https://github.com/nlohmann/json/releases/download/v3.11.3/json.tar.xz
        URL_HASH SHA256=d6c65aca6b1ed68e7a182f4757257b107ae403032760ed6ef121c9d55e81757d
        DOWNLOAD_EXTRACT_TIMESTAMP TRUE
    )
    FetchContent_MakeAvailable(json)
endif()

add_library(
    mvs_core STATIC
    mvs/common.cpp
    mvs/cluster.cpp
    mvs/dfg.cpp
    mvs/graph.cpp
    mvs/io.cpp
    mvs/mvs.cpp
    mvs/vs.cpp
)
target_include_directories(mvs_core PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/mvs")
target_link_libraries(mvs_core PUBLIC nlohmann_json::nlohmann_json)
target_compile_options(
    mvs_core PRIVATE
    -Wall
    -Wextra
    -Wno-sign-compare
    -Wno-unused-function
)

nanobind_add_module(
    _native
    NB_STATIC
    src/cpp/bindings.cpp
)
target_include_directories(_native PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/mvs")
target_link_libraries(_native PRIVATE mvs_core nlohmann_json::nlohmann_json)

install(TARGETS _native LIBRARY DESTINATION mvs)
