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

#ifndef SCANN_UTILS_IO_NPY_H_
#define SCANN_UTILS_IO_NPY_H_

#include <cstddef>
#include <fstream>
#include <string>
#include <utility>
#include <vector>

#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"
#include "scann/data_format/dataset.h"
#include "scann/oss_wrappers/scann_status.h"
#include "scann/utils/common.h"
#include "scann/utils/io_oss_wrapper.h"
#include "scann/utils/types.h"

namespace research_scann {

template <typename T>
std::string numpy_type_name();

template <typename T>
Status SpanToNumpy(absl::string_view filename, ConstSpan<T> data,
                   ConstSpan<size_t> dim_sizes = {},
                   DimensionIndex last_dim = kInvalidDimension) {
  std::string shape_str = "(";
  size_t dim_prod = 1;
  for (size_t dim_size : dim_sizes) {
    dim_prod *= dim_size;
    shape_str += std::to_string(dim_size) + ",";
  }
  if (last_dim != kInvalidDimension) {
    shape_str += std::to_string(last_dim) + ",)";
  } else {
    if (dim_prod == 0 || data.size() % dim_prod != 0)
      return InvalidArgumentError(
          "Size of data isn't compatible with given shape");
    shape_str += std::to_string(data.size() / dim_prod) + ",)";
  }

  if (shape_str.size() > 65000)
    return InvalidArgumentError("Shape string is too large for npy format: " +
                                shape_str);

  std::string pt1("\x93NUMPY\x01\x00  ", 10);
  std::string dict =
      absl::StrFormat("{'descr':%s, 'fortran_order':False, 'shape':%s}",
                      numpy_type_name<T>(), shape_str);
  while ((pt1.size() + dict.size()) % 64 != 63) dict += ' ';
  dict += '\n';
  pt1[8] = dict.size() % 256;
  pt1[9] = dict.size() / 256;
  const std::string header = pt1 + dict;

  OpenSourceableFileWriter writer(filename);
  SCANN_RETURN_IF_ERROR(
      writer.Write(ConstSpan<char>(header.data(), header.size())));

  const char* ptr = reinterpret_cast<const char*>(data.data());
  return writer.Write(ConstSpan<char>(ptr, data.size() * sizeof(T)));
}

template <typename T>
Status VectorToNumpy(absl::string_view filename, const vector<T>& data,
                     const vector<size_t>& dim_sizes = {}) {
  return SpanToNumpy(filename, ConstSpan<T>(data.data(), data.size()),
                     ConstSpan<size_t>(dim_sizes.data(), dim_sizes.size()));
}

template <typename T>
Status DatasetToNumpy(absl::string_view filename, const DenseDataset<T>& data) {
  return SpanToNumpy(filename, data.data(), {data.size()},
                     data.dimensionality());
}

namespace npy_internal {

// scann-core: opens `filename` as a .npy file of dtype `expected_descr` (as
// numpy_type_name returns it, e.g. "'<f4'") and validates it: magic,
// version, header length, a well-formed header dict, dtype kind, word size
// and byte order, C order, a shape whose element and byte counts don't
// overflow, and a data size equal to the rest of the file. Returns the
// shape, with `in` positioned at the data. Upstream parsed the header with
// cnpy, which reads past its buffer for a bad header length, throws
// std::out_of_range for a dimension above INT_MAX, and ignores the dtype's
// kind and byte order (an int32 or big-endian file was read as float32);
// nothing compared the shape with the file size.
StatusOr<std::vector<size_t>> OpenNpy(absl::string_view filename,
                                      absl::string_view expected_descr,
                                      size_t word_size, std::ifstream& in);

}  // namespace npy_internal

template <typename T>
StatusOr<pair<std::vector<T>, std::vector<size_t>>> NumpyToVectorAndShape(
    absl::string_view filename) {
  std::ifstream in;
  SCANN_ASSIGN_OR_RETURN(
      std::vector<size_t> shape,
      npy_internal::OpenNpy(filename, numpy_type_name<T>(), sizeof(T), in));
  size_t total_size = 1;
  for (size_t s : shape) total_size *= s;
  vector<T> buffer(total_size);
  if (total_size > 0 && !in.read(reinterpret_cast<char*>(buffer.data()),
                                 total_size * sizeof(T)))
    return InternalError(absl::StrCat("I/O error reading ", filename));
  return std::make_pair(std::move(buffer), std::move(shape));
}

}  // namespace research_scann

#endif
