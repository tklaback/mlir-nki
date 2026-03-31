cmake_minimum_required(VERSION 3.20)
project(my-project)

find_package(MLIR REQUIRED CONFIG)
list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")

include(TableGen)
include(AddMLIR)
include(AddLLVM)

add_subdirectory(include)
add_subdirectory(lib)
add_subdirectory(tools)