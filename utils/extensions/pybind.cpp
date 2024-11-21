#include <torch/extension.h>

#include "cpu/radius_neighbors/radius_neighbors.h"
#include "cpu/grid_subsampling/grid_subsampling.h"
#include "cpu/radius_filter/radius_filter.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  // CPU extensions
  m.def(
    "radius_neighbors",
    &radius_neighbors,
    "Radius neighbors (CPU)"
  );
  m.def(
    "grid_subsampling",
    &grid_subsampling,
    "Grid subsampling (CPU)"
  );
  m.def(
    "radius_filter",
    &radius_filter,
    "radius filter (CPU)"
  );
}
