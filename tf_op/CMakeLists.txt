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

# The optional TensorFlow custom op (-DSCANN_BUILD_TF_OP=ON; source-only,
# not part of the wheel). See docs/tensorflow.md.
#
#   <build>/python/scann_tf_ops/_scann_tf_ops.so   the op library
#   <build>/python/scann_tf_ops/__init__.py        its Python API
#
# It is built against the TensorFlow installed in Python_EXECUTABLE, but
# only against its C API (tensorflow/c/kernels.h, ops.h): no TensorFlow C++
# header, so none of TensorFlow's abseil or protobuf. scann-core and all of
# its dependencies are linked in statically and kept private: the library
# exports no symbols (a `local: *` version script, --exclude-libs=ALL,
# hidden visibility), and imports only TensorFlow's C functions (TF_*) from
# libtensorflow_framework.so.2, which it lists as NEEDED without an rpath:
# TensorFlow has always loaded it before tf.load_op_library() runs.

if(NOT CMAKE_SYSTEM_NAME STREQUAL "Linux")
  message(FATAL_ERROR "SCANN_BUILD_TF_OP: the TensorFlow op builds on Linux only (it uses a GNU ld version script).")
endif()
if(NOT SCANN_BUILD_PYTHON)
  message(FATAL_ERROR "SCANN_BUILD_TF_OP needs SCANN_BUILD_PYTHON=ON (its Python package builds on the scann package).")
endif()

# TensorFlow's headers and libtensorflow_framework.so.2, from the Python
# the package is built for.
execute_process(
  COMMAND "${CMAKE_COMMAND}" -E env TF_CPP_MIN_LOG_LEVEL=3
          "${Python_EXECUTABLE}" -c
          "import tensorflow as tf; print(tf.__version__); print(tf.sysconfig.get_include()); print(tf.sysconfig.get_lib())"
  OUTPUT_VARIABLE _scann_tf_out
  ERROR_VARIABLE _scann_tf_err
  RESULT_VARIABLE _scann_tf_rc
  OUTPUT_STRIP_TRAILING_WHITESPACE)
if(NOT _scann_tf_rc EQUAL 0)
  message(FATAL_ERROR "SCANN_BUILD_TF_OP: can't import tensorflow with ${Python_EXECUTABLE}:\n${_scann_tf_err}")
endif()
string(REPLACE "\n" ";" _scann_tf_out "${_scann_tf_out}")
list(GET _scann_tf_out -3 SCANN_TF_VERSION)
list(GET _scann_tf_out -2 SCANN_TF_INCLUDE)
list(GET _scann_tf_out -1 SCANN_TF_LIBDIR)
set(SCANN_TF_FRAMEWORK "${SCANN_TF_LIBDIR}/libtensorflow_framework.so.2")
if(NOT EXISTS "${SCANN_TF_INCLUDE}/tensorflow/c/kernels.h" OR NOT EXISTS "${SCANN_TF_FRAMEWORK}")
  message(FATAL_ERROR "SCANN_BUILD_TF_OP: TensorFlow ${SCANN_TF_VERSION} has no tensorflow/c/kernels.h under ${SCANN_TF_INCLUDE} or no ${SCANN_TF_FRAMEWORK}")
endif()
message(STATUS "scann-core: TensorFlow op built against TensorFlow ${SCANN_TF_VERSION} (${SCANN_TF_LIBDIR})")

set(SCANN_TF_PY_OUT "${PROJECT_BINARY_DIR}/python/scann_tf_ops")

# All of scann-core's objects (for the static-initializer registrations,
# see ../CMakeLists.txt), and its static dependencies as the single archive
# libscann_core_deps.a (built with -fPIC, CMAKE_POSITION_INDEPENDENT_CODE),
# linked explicitly in this order: libtensorflow_framework.so.2 also defines
# abseil and protobuf symbols, so it must come after every archive, or the
# linker resolves scann-core's abseil/protobuf references to TensorFlow's
# copies instead of pulling in its own. (Linking scann_core_objects as a
# target would list the dependency archives after the TensorFlow library.)
if(NOT SCANN_BUILD_STATIC)
  message(FATAL_ERROR "SCANN_BUILD_TF_OP needs SCANN_BUILD_STATIC=ON (it links libscann_core_deps.a).")
endif()
add_library(scann_tf_ops MODULE scann_tf_ops.cc $<TARGET_OBJECTS:scann_core_objects>)
add_dependencies(scann_tf_ops scann_core_deps_bundle)
target_link_libraries(scann_tf_ops PRIVATE
  "$<COMPILE_ONLY:scann_core_deps>"
  "${PROJECT_BINARY_DIR}/libscann_core_deps.a"
  ${SCANN_CORE_SYSTEM_LIBS}
  "${SCANN_TF_FRAMEWORK}")
# -idirafter: TensorFlow's include directory also holds its own abseil,
# protobuf and Eigen headers, which must never shadow scann-core's. Searched
# after every -I/-isystem directory, it only supplies tensorflow/c/ and tsl/.
target_compile_options(scann_tf_ops PRIVATE
  -fvisibility=hidden -fvisibility-inlines-hidden
  "SHELL:-idirafter ${SCANN_TF_INCLUDE}")
target_link_options(scann_tf_ops PRIVATE
  "LINKER:--version-script=${CMAKE_CURRENT_SOURCE_DIR}/exports.map"
  "LINKER:--exclude-libs,ALL"
  "LINKER:--no-undefined")
set_target_properties(scann_tf_ops PROPERTIES
  PREFIX ""
  OUTPUT_NAME _scann_tf_ops
  LIBRARY_OUTPUT_DIRECTORY "${SCANN_TF_PY_OUT}"
  # No rpath to the TensorFlow of the build machine.
  SKIP_BUILD_RPATH ON
  LINK_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/exports.map")

# The scann_tf_ops Python package, next to the scann package:
#   <build>/python/scann_tf_ops/{__init__.py,_scann_tf_ops.so}
add_custom_command(
  OUTPUT "${SCANN_TF_PY_OUT}/__init__.py"
  COMMAND "${CMAKE_COMMAND}" -E copy
          "${CMAKE_CURRENT_SOURCE_DIR}/python/scann_tf_ops/__init__.py"
          "${SCANN_TF_PY_OUT}/__init__.py"
  DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/python/scann_tf_ops/__init__.py"
  VERBATIM)
add_custom_target(scann_tf_ops_python ALL
  DEPENDS "${SCANN_TF_PY_OUT}/__init__.py" scann_tf_ops scann_python_package)

if(SCANN_BUILD_TESTS AND CMAKE_NM AND CMAKE_READELF)
  # Exports nothing, imports only TF_* C functions and the C/C++ runtime,
  # needs libtensorflow_framework.so.2 without an rpath.
  add_test(NAME tf_op_symbols
           COMMAND "${Python_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/tests/check_symbols.py"
                   "$<TARGET_FILE:scann_tf_ops>" "${CMAKE_NM}" "${CMAKE_READELF}")
  set_tests_properties(tf_op_symbols PROPERTIES TIMEOUT 60)
endif()

if(SCANN_BUILD_TESTS AND SCANN_PYTHON_RUNTIME_OK)
  # scann_tf_ops against the pybind searcher: parity, build-once, stale
  # variables, SavedModel in a fresh process, errors, concurrency. Skipped
  # (exit code 77) when TensorFlow isn't importable.
  function(scann_add_tf_op_test name script)
    add_test(NAME ${name}
             COMMAND "${CMAKE_COMMAND}" -E env "PYTHONPATH=${PROJECT_BINARY_DIR}/python"
                     "${Python_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/${script}")
    set_tests_properties(${name} PROPERTIES TIMEOUT 600 SKIP_RETURN_CODE 77)
  endfunction()
  scann_add_tf_op_test(python_tf_ops tests/test_scann_tf_ops.py)
endif()
