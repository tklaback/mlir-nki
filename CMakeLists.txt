cmake_minimum_required(VERSION 3.20.0)
project(nki)

set(CMAKE_CXX_STANDARD 17)

find_package(LLVM REQUIRED CONFIG)
find_package(MLIR REQUIRED CONFIG)


set(LLVM_RUNTIME_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/bin)
set(LLVM_LIBRARY_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/lib)

list(APPEND CMAKE_MODULE_PATH ${MLIR_INCLUDE_DIR})
list(APPEND CMAKE_MODULE_PATH ${LLVM_CMAKE_DIR})
list(APPEND CMAKE_MODULE_PATH ${MLIR_CMAKE_DIR})

include(TableGen)
include(AddLLVM)
include(AddMLIR)

include_directories(${CMAKE_CURRENT_BINARY_DIR})
include_directories(${CMAKE_CURRENT_BINARY_DIR}/include/)
include_directories(${MLIR_INCLUDE_DIRS})

add_subdirectory(include)

add_mlir_library(NKIDialect
  lib/Dialect.cpp

  LINK_LIBS PUBLIC
  MLIRIR
)

add_dependencies(NKIDialect NKIOpsIncGen)

target_include_directories(NKIDialect PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)

add_library(NKITransforms
  lib/Transforms/NKIToPython.cpp
)

add_dependencies(NKITransforms NKIOpsIncGen)

target_include_directories(NKITransforms PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)

add_executable(nki-opt
  tools/nki-opt.cpp
)

add_dependencies(nki-opt NKIOpsIncGen)

target_include_directories(nki-opt PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)

target_link_libraries(nki-opt PRIVATE
  NKIDialect
  MLIROptLib
  MLIRFuncDialect
  MLIRSupport
)