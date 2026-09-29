// Copyright 2026 The Google Research Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
// Modified in 2026 by Elias Benali (@ebenali) and TheCleaners for
// scann-core (a derived work of ScaNN, not an official Google product);
// see NOTICE.

#include <cstdint>
#include <string>

#include "absl/types/optional.h"
#include "pybind11/pybind11.h"
#include "pybind11/stl.h"
#include "scann/scann_ops/cc/scann_npy.h"

// scann-core: mod_gil_not_used() lets free-threaded Python (3.14t, ...) keep
// the GIL disabled when this module is imported; ScannNumpy does its own
// locking (see scann_npy.h).
PYBIND11_MODULE(scann_pybind, py_module, pybind11::mod_gil_not_used()) {
  py_module.doc() = "pybind11 wrapper for ScaNN";
  pybind11::class_<research_scann::ScannNumpy>(py_module, "ScannNumpy")
      .def(pybind11::init<const std::string&, const std::string&>())
      .def(pybind11::init<const research_scann::np_row_major_arr<float>&,
                          const std::string&, int>())
      .def("search", &research_scann::ScannNumpy::Search)
      .def("search_batched", &research_scann::ScannNumpy::SearchBatched)

      // scann-core: a float32 C-contiguous 2-D array is read in place
      // (noconvert: anything else, e.g. a list of rows, falls through to
      // the list-of-rows overload, as before).
      .def("upsert",
           pybind11::overload_cast<
               std::vector<std::optional<research_scann::DatapointIndex>>,
               const research_scann::np_row_major_arr<float>&, int>(
               &research_scann::ScannNumpy::Upsert),
           pybind11::arg("indices"), pybind11::arg("vectors").noconvert(),
           pybind11::arg("batch_size") = 256)
      .def("upsert",
           pybind11::overload_cast<
               std::vector<std::optional<research_scann::DatapointIndex>>,
               std::vector<research_scann::np_row_major_arr<float>>&, int>(
               &research_scann::ScannNumpy::Upsert),
           pybind11::arg("indices"), pybind11::arg("vectors"),
           pybind11::arg("batch_size") = 256)
      .def("delete", &research_scann::ScannNumpy::Delete)
      .def("rebalance", &research_scann::ScannNumpy::Rebalance)
      .def_static("suggest_autopilot",
                  &research_scann::ScannNumpy::SuggestAutopilot)
      .def("size", &research_scann::ScannNumpy::Size)
      .def("reserve", &research_scann::ScannNumpy::Reserve)
      .def("set_num_threads", &research_scann::ScannNumpy::SetNumThreads)
      .def("config", &research_scann::ScannNumpy::Config)
      .def("serialize", &research_scann::ScannNumpy::Serialize,
           pybind11::arg("path"), pybind11::arg("relative_path") = false,
           pybind11::arg("docids_pkl") = pybind11::none())
      .def("get_health_stats", &research_scann::ScannNumpy::GetHealthStats)
      .def("initialize_health_stats",
           &research_scann::ScannNumpy::InitializeHealthStats);
}
