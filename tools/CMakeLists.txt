add_llvm_executable(nki-opt nki-opt.cpp)

target_link_libraries(nki-opt PRIVATE
  NKIIR
  NKITransforms
  MLIROptLib
  MLIRPass
  MLIRFuncDialect
)