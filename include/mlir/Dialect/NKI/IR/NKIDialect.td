// Provide a definition of the 'toy' dialect in the ODS framework so that we
// can define our operations.
def NKI_Dialect : Dialect {
  // The namespace of our dialect, this corresponds 1-1 with the string we
  // provided in `ToyDialect::getDialectNamespace`.
  let name = "nki";

  // A short one-line summary of our dialect.
  let summary = "";

  // A much longer description of our dialect.
  let description = [{}];

  // The C++ namespace that the dialect class definition resides in.
  let cppNamespace = "nki";
}