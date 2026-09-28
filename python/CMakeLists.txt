# Copyright 2026 Elias Benali (@ebenali) and TheCleaners.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# The `scann` Python package, laid out exactly like the upstream wheel so that
# code written against `scann.scann_ops_pybind` runs unchanged:
#
#   <build>/python/scann/__init__.py
#   <build>/python/scann/scann_ops/py/{scann_builder,scann_ops_pybind,
#                                      scann_ops_pybind_backcompat}.py
#   <build>/python/scann/scann_ops/cc/python/scann_pybind*.so
#   <build>/python/scann/**/<name>_pb2.py        (the same 20 as the wheel)
#
# The three .py files under python/scann/scann_ops/py/ are upstream's;
# scann_ops_pybind.py carries the fixes listed in NOTICE. scann/__init__.py
# is scann-core's: it omits the TensorFlow op import (scann-core does not
# build the TF custom op, and importing TensorFlow just to discover that
# would defeat the point of decoupling). scann/tf.py (adapted from upstream's
# scann_ops.py) wraps the pybind searcher for TensorFlow code, and
# scann/torch.py for PyTorch; only an explicit `import scann.tf` or
# `import scann.torch` imports TensorFlow or PyTorch.

set(SCANN_PY_OUT "${PROJECT_BINARY_DIR}/python")

# scann_npy.cc (the ScannNumpy facade) and scann_pybind.cc link the core's
# object files directly -- all of them, so the static-initializer distance
# measure registrations are always present (see ../CMakeLists.txt).
pybind11_add_module(scann_pybind
  "${SCANN_SRC_DIR}/scann/scann_ops/cc/scann_npy.cc"
  "${SCANN_SRC_DIR}/scann/scann_ops/cc/python/scann_pybind.cc")
target_link_libraries(scann_pybind PRIVATE scann_core_objects)
set_target_properties(scann_pybind PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY "${SCANN_PY_OUT}/scann/scann_ops/cc/python")

# Exactly the protos the wheel ships _pb2 modules for (its py_proto_library
# targets); the remaining five are C++-only upstream too.
set(SCANN_PY_PROTOS
  scann/data_format/features.proto
  scann/partitioning/kmeans_tree_partitioner.proto
  scann/partitioning/linear_projection_tree.proto
  scann/partitioning/partitioner.proto
  scann/proto/auto_tuning.proto
  scann/proto/brute_force.proto
  scann/proto/centers.proto
  scann/proto/crowding.proto
  scann/proto/disjoint_restrict_token.proto
  scann/proto/distance_measure.proto
  scann/proto/exact_reordering.proto
  scann/proto/hash.proto
  scann/proto/input_output.proto
  scann/proto/metadata.proto
  scann/proto/min_distance.proto
  scann/proto/partitioning.proto
  scann/proto/projection.proto
  scann/proto/scann.proto
  scann/scann_ops/scann_assets.proto
  scann/trees/kmeans_tree/kmeans_tree.proto)
list(TRANSFORM SCANN_PY_PROTOS PREPEND "${SCANN_SRC_DIR}/")
scann_generate_python_protos(SCANN_PB2_FILES
  PROTO_ROOT "${SCANN_SRC_DIR}"
  OUT_DIR "${SCANN_PY_OUT}"
  PROTOS ${SCANN_PY_PROTOS})

file(GLOB_RECURSE SCANN_PY_SOURCES CONFIGURE_DEPENDS
  RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}/scann/*.py")
set(SCANN_PY_COPIED "")
foreach(rel ${SCANN_PY_SOURCES})
  add_custom_command(
    OUTPUT "${SCANN_PY_OUT}/${rel}"
    COMMAND "${CMAKE_COMMAND}" -E copy "${CMAKE_CURRENT_SOURCE_DIR}/${rel}" "${SCANN_PY_OUT}/${rel}"
    DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/${rel}"
    VERBATIM)
  list(APPEND SCANN_PY_COPIED "${SCANN_PY_OUT}/${rel}")
endforeach()

configure_file("${CMAKE_CURRENT_SOURCE_DIR}/scann/version.py.in"
               "${SCANN_PY_OUT}/scann/version.py" @ONLY)

add_custom_target(scann_python_package ALL
  DEPENDS ${SCANN_PB2_FILES} ${SCANN_PY_COPIED} scann_pybind)

# pip install / wheel builds (scikit-build-core, see pyproject.toml): the
# package is assembled in ${SCANN_PY_OUT}/scann and installed as is.
if(SKBUILD)
  install(DIRECTORY "${SCANN_PY_OUT}/scann" DESTINATION .
          PATTERN "__pycache__" EXCLUDE
          PATTERN "*.pyc" EXCLUDE)
endif()
