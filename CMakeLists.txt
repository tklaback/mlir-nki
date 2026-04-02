cmake_minimum_required(VERSION 3.20)
project(nki-dialect)

find_package(MLIR REQUIRED CONFIG HINTS ${MLIR_DIR})
find_package(AIR REQUIRED CONFIG HINTS ${AIR_DIR})

list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")

include(TableGen)
include(AddMLIR)
include(AddLLVM)

include_directories(${MLIR_INCLUDE_DIRS})
include_directories(${LLVM_INCLUDE_DIRS})
include_directories(${AIR_INCLUDE_DIRS})
# AIR's generated .h.inc files live in the build tree, not the install tree.
# Pass -DAIR_BUILD_DIR=<path/to/mlir-air/build> at cmake configure time.
if(AIR_BUILD_DIR)
  include_directories(${AIR_BUILD_DIR}/mlir/include)
endif()
include_directories(${PROJECT_SOURCE_DIR}/include)
include_directories(${PROJECT_BINARY_DIR}/include)  # for generated .inc files

add_definitions(${LLVM_DEFINITIONS})

add_subdirectory(include)
add_subdirectory(lib)
add_subdirectory(tools)