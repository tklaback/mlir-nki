add_llvm_executable(nki-opt nki-opt.cpp)

target_link_libraries(nki-opt PRIVATE
  AIRDialect
  NKIIR
  NKITransforms
  MLIROptLib
  MLIRPass
  MLIRFuncDialect
  MLIRAffineDialect
  MLIRArithDialect
  MLIRTensorDialect
  MLIRSCFDialect
  MLIRLinalgDialect
  MLIRMemRefDialect
)