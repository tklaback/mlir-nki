#ifndef NKI_TRANSFORMS_PASSES_H
#define NKI_TRANSFORMS_PASSES_H

#include <memory>

namespace nki {
std::unique_ptr<mlir::Pass> createNKIToPythonPass();
}

#endif