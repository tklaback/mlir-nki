cmake_minimum_required(VERSION 3.20.0)
project(toy)

add_custom_target(Toy)

macro(add_toy_chapter name)
  add_dependencies(Toy ${name})
  add_llvm_example(${name} ${ARGN})
endmacro(add_toy_chapter name)
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

include_directories(include)
include_directories(${CMAKE_CURRENT_BINARY_DIR})
include_directories(${CMAKE_CURRENT_BINARY_DIR}/include/)
include_directories(${MLIR_INCLUDE_DIRS})

add_subdirectory(include)
unset(LLVM_TARGET_DEFINITIONS)

# Combine tablegen
set(LLVM_TARGET_DEFINITIONS mlir/ToyCombine.td)
mlir_tablegen(ToyCombine.inc -gen-rewriters -I${MLIR_INCLUDE_DIR})
add_public_tablegen_target(ToyCombineIncGen)
unset(LLVM_TARGET_DEFINITIONS)

add_llvm_executable(toyc
  toyc.cpp
  parser/AST.cpp
  mlir/MLIRGen.cpp
  mlir/Dialect.cpp
  mlir/ToyCombine.cpp
)

add_dependencies(toyc ToyOpsIncGen ToyCombineIncGen)

target_link_libraries(toyc PRIVATE
  MLIRAnalysis
  MLIRFunctionInterfaces
  MLIRIR
  MLIRParser
  MLIRPass
  MLIRSideEffectInterfaces
  MLIRTransforms
  LLVMSupport
)