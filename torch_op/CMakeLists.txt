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

# The native PyTorch op (-DSCANN_BUILD_TORCH_OP=ON; the scann-core-torch
# wheel, built from torch_op/pyproject.toml). See docs/integrations.md.
#
#   <build>/python/scann_torch_ops/_scann_torch_ops.so   the op library
#   <build>/python/scann_torch_ops/{__init__,_version}.py its Python package
#
# It is built against the headers of the torch installed in
# SCANN_TORCH_PYTHON (default: Python_EXECUTABLE), but only against
# LibTorch's stable ABI (torch/csrc/stable, torch/headeronly), targeting
# torch 2.10: no c10/ATen/torch C++ symbol, so the library runs on any
# torch >= 2.10, CPU, CUDA or ROCm build, and on any Python (it uses no
# Python C API). scann-core and all of its dependencies are linked in
# statically and kept private: the library exports no symbols (a
# `local: *` version script, --exclude-libs=ALL, hidden visibility), and
# imports only the stable C shim functions (aoti_torch_*, torch_*) from
# libtorch_cpu.so, which it lists as NEEDED without an rpath: `import
# torch` has loaded it before the library is.

if(NOT CMAKE_SYSTEM_NAME STREQUAL "Linux")
  message(FATAL_ERROR "SCANN_BUILD_TORCH_OP: the PyTorch op builds on Linux only (it uses a GNU ld version script).")
endif()
if(NOT SCANN_BUILD_STATIC)
  message(FATAL_ERROR "SCANN_BUILD_TORCH_OP needs SCANN_BUILD_STATIC=ON (it links libscann_core_deps.a).")
endif()

set(SCANN_TORCH_PYTHON "" CACHE FILEPATH
  "Python whose torch supplies the headers and libraries the op is built against (default: Python_EXECUTABLE)")
if(SCANN_TORCH_PYTHON)
  set(_scann_torch_py "${SCANN_TORCH_PYTHON}")
elseif(Python_EXECUTABLE)
  set(_scann_torch_py "${Python_EXECUTABLE}")
else()
  find_package(Python 3.10 REQUIRED COMPONENTS Interpreter)
  set(_scann_torch_py "${Python_EXECUTABLE}")
endif()

# torch's include and lib directories, found from its package (not its CMake
# config, which sets up the non-stable C++ API). Importing torch isn't
# needed: its location is enough.
execute_process(
  COMMAND "${_scann_torch_py}" -c
          "import importlib.util as u, os; s = u.find_spec('torch'); d = os.path.dirname(s.origin); v = {}; exec(open(os.path.join(d, 'version.py')).read(), v); print(v['__version__']); print(os.path.join(d, 'include')); print(os.path.join(d, 'lib'))"
  OUTPUT_VARIABLE _scann_torch_out
  ERROR_VARIABLE _scann_torch_err
  RESULT_VARIABLE _scann_torch_rc
  OUTPUT_STRIP_TRAILING_WHITESPACE)
if(NOT _scann_torch_rc EQUAL 0)
  message(FATAL_ERROR "SCANN_BUILD_TORCH_OP: can't find torch with ${_scann_torch_py}:\n${_scann_torch_err}")
endif()
string(REPLACE "\n" ";" _scann_torch_out "${_scann_torch_out}")
list(GET _scann_torch_out -3 SCANN_TORCH_VERSION)
list(GET _scann_torch_out -2 SCANN_TORCH_INCLUDE)
list(GET _scann_torch_out -1 SCANN_TORCH_LIBDIR)
string(REGEX MATCH "^[0-9]+\\.[0-9]+" _scann_torch_mm "${SCANN_TORCH_VERSION}")
if(_scann_torch_mm VERSION_LESS 2.10)
  message(FATAL_ERROR "SCANN_BUILD_TORCH_OP needs torch >= 2.10 (its stable ABI); ${_scann_torch_py} has ${SCANN_TORCH_VERSION}.")
endif()
foreach(_f include/torch/csrc/stable/library.h lib/libtorch_cpu.so)
  get_filename_component(_d "${SCANN_TORCH_INCLUDE}" DIRECTORY)
  if(NOT EXISTS "${_d}/${_f}")
    message(FATAL_ERROR "SCANN_BUILD_TORCH_OP: torch ${SCANN_TORCH_VERSION} has no ${_d}/${_f}")
  endif()
endforeach()
message(STATUS "scann-core: PyTorch op built against torch ${SCANN_TORCH_VERSION} (${SCANN_TORCH_LIBDIR}), stable ABI targeting 2.10")

set(SCANN_TORCH_PY_OUT "${PROJECT_BINARY_DIR}/python/scann_torch_ops")

# Linked like the TensorFlow op (see ../tf_op/CMakeLists.txt): all of
# scann-core's objects (for the static-initializer registrations), then its
# dependencies as the single archive libscann_core_deps.a, then the system
# libraries and the C++ standard library, and libtorch_cpu.so last (only
# for the stable C shim). torch's libraries export symbols that aren't
# theirs: libtorch_cpu.so over 200 of protobuf's, and it and libc10.so some
# of libstdc++'s (std::filesystem, std::string members), unversioned. Listed
# before the archives, the linker would bind scann-core's references to
# torch's protobuf instead of pulling in its own; listed before libstdc++
# (which the compiler driver otherwise adds at the very end), it binds the
# C++ runtime references to torch's copies, unversioned, and makes
# libc10.so a dependency. torch_op_symbols checks the result.
if(CMAKE_CXX_FLAGS MATCHES "-stdlib=libc\\+\\+")
  set(_scann_cxx_runtime "")
else()
  set(_scann_cxx_runtime stdc++)
endif()
add_library(scann_torch_ops MODULE scann_torch_ops.cc $<TARGET_OBJECTS:scann_core_objects>)
add_dependencies(scann_torch_ops scann_core_deps_bundle)
target_link_libraries(scann_torch_ops PRIVATE
  "$<COMPILE_ONLY:scann_core_deps>"
  "${PROJECT_BINARY_DIR}/libscann_core_deps.a"
  ${SCANN_CORE_SYSTEM_LIBS}
  ${_scann_cxx_runtime}
  "${SCANN_TORCH_LIBDIR}/libtorch_cpu.so")
# TORCH_TARGET_VERSION: only the stable API of torch 2.10 (and nothing
# newer) is available, whatever the headers' version.
# -idirafter: torch's include directory also holds other libraries'
# headers (google/protobuf, pybind11, ...), which must never shadow
# scann-core's; searched after every -I/-isystem directory, it only
# supplies torch/.
target_compile_definitions(scann_torch_ops PRIVATE
  TORCH_TARGET_VERSION=0x020a000000000000)
target_compile_options(scann_torch_ops PRIVATE
  -fvisibility=hidden -fvisibility-inlines-hidden
  "SHELL:-idirafter ${SCANN_TORCH_INCLUDE}")
target_link_options(scann_torch_ops PRIVATE
  "LINKER:--version-script=${CMAKE_CURRENT_SOURCE_DIR}/exports.map"
  "LINKER:--exclude-libs,ALL"
  "LINKER:--no-undefined")
set_target_properties(scann_torch_ops PROPERTIES
  PREFIX ""
  OUTPUT_NAME _scann_torch_ops
  LIBRARY_OUTPUT_DIRECTORY "${SCANN_TORCH_PY_OUT}"
  # No rpath to the torch of the build machine.
  SKIP_BUILD_RPATH ON
  LINK_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/exports.map")

# The scann_torch_ops Python package, next to the scann package:
#   <build>/python/scann_torch_ops/{__init__.py,_version.py,_scann_torch_ops.so}
configure_file("${CMAKE_CURRENT_SOURCE_DIR}/python/scann_torch_ops/_version.py.in"
               "${SCANN_TORCH_PY_OUT}/_version.py" @ONLY)
add_custom_command(
  OUTPUT "${SCANN_TORCH_PY_OUT}/__init__.py"
  COMMAND "${CMAKE_COMMAND}" -E copy
          "${CMAKE_CURRENT_SOURCE_DIR}/python/scann_torch_ops/__init__.py"
          "${SCANN_TORCH_PY_OUT}/__init__.py"
  DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/python/scann_torch_ops/__init__.py"
  VERBATIM)
add_custom_target(scann_torch_ops_python ALL
  DEPENDS "${SCANN_TORCH_PY_OUT}/__init__.py" scann_torch_ops)

# The scann-core-torch wheel (scikit-build-core, see pyproject.toml here):
# just the scann_torch_ops package, with the license files, as the install
# component scann_torch_ops, the only one the wheel installs (Eigen's own
# install rules would add its headers).
if(SKBUILD)
  install(DIRECTORY "${SCANN_TORCH_PY_OUT}" DESTINATION .
          COMPONENT scann_torch_ops
          PATTERN "__pycache__" EXCLUDE
          PATTERN "*.pyc" EXCLUDE)
  install(FILES "${PROJECT_SOURCE_DIR}/LICENSE" "${PROJECT_SOURCE_DIR}/NOTICE"
          DESTINATION scann_torch_ops COMPONENT scann_torch_ops)
endif()

if(SCANN_BUILD_TESTS AND CMAKE_NM AND CMAKE_READELF)
  # Exports nothing; imports only the stable C shim and the C/C++ runtime;
  # needs libtorch_cpu.so without an rpath, and no libpython.
  find_package(Python 3.10 COMPONENTS Interpreter)
  add_test(NAME torch_op_symbols
           COMMAND "${Python_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/tests/check_symbols.py"
                   "$<TARGET_FILE:scann_torch_ops>" "${CMAKE_NM}" "${CMAKE_READELF}"
                   "${SCANN_TORCH_LIBDIR}")
  set_tests_properties(torch_op_symbols PROPERTIES TIMEOUT 60)
endif()
