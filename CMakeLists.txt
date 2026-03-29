cmake_minimum_required(VERSION 3.20)
project(mlir-nki)

set(CMAKE_CXX_STANDARD 17)

find_package(LLVM REQUIRED CONFIG)
find_package(MLIR REQUIRED CONFIG)

set(LLVM_RUNTIME_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/bin)
set(LLVM_LIBRARY_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/lib)

list(APPEND CMAKE_MODULE_PATH ${LLVM_CMAKE_DIR})
list(APPEND CMAKE_MODULE_PATH ${MLIR_CMAKE_DIR})

include(TableGen)
include(AddLLVM)
include(AddMLIR)

add_subdirectory(lib)