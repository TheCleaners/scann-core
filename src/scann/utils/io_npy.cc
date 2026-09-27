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

#include "scann/utils/io_npy.h"

#include <cstdint>
#include <cstring>
#include <limits>
#include <string>
#include <type_traits>

#include "absl/strings/match.h"
#include "absl/strings/str_cat.h"

namespace research_scann {

// scann-core: a strict .npy header reader replacing cnpy::parse_npy_header;
// see io_npy.h.
namespace npy_internal {
namespace {

constexpr size_t kMaxHeaderLen = size_t{1} << 20;

bool HostIsLittleEndian() {
  const uint16_t one = 1;
  unsigned char first;
  std::memcpy(&first, &one, 1);
  return first == 1;
}

// Parses the Python literal dict of a .npy header, e.g.
// {'descr': '<f4', 'fortran_order': False, 'shape': (3, 4), }
class HeaderParser {
 public:
  explicit HeaderParser(absl::string_view s) : s_(s) {}

  void SkipSpace() {
    while (pos_ < s_.size() && (s_[pos_] == ' ' || s_[pos_] == '\t' ||
                                s_[pos_] == '\n' || s_[pos_] == '\r'))
      ++pos_;
  }
  bool Consume(char c) {
    SkipSpace();
    if (pos_ < s_.size() && s_[pos_] == c) {
      ++pos_;
      return true;
    }
    return false;
  }
  bool ConsumeWord(absl::string_view word) {
    SkipSpace();
    if (!absl::StartsWith(s_.substr(pos_), word)) return false;
    pos_ += word.size();
    return true;
  }
  bool ParseString(std::string* out) {
    SkipSpace();
    if (pos_ >= s_.size() || (s_[pos_] != '\'' && s_[pos_] != '"'))
      return false;
    const size_t end = s_.find(s_[pos_], pos_ + 1);
    if (end == absl::string_view::npos) return false;
    *out = std::string(s_.substr(pos_ + 1, end - pos_ - 1));
    pos_ = end + 1;
    return true;
  }
  bool ParseUint(uint64_t* out) {
    SkipSpace();
    const size_t start = pos_;
    uint64_t v = 0;
    while (pos_ < s_.size() && s_[pos_] >= '0' && s_[pos_] <= '9') {
      const uint64_t d = s_[pos_] - '0';
      if (v > (std::numeric_limits<uint64_t>::max() - d) / 10) return false;
      v = v * 10 + d;
      ++pos_;
    }
    *out = v;
    return pos_ > start;
  }
  bool AtEnd() {
    SkipSpace();
    return pos_ == s_.size();
  }

 private:
  absl::string_view s_;
  size_t pos_ = 0;
};

}  // namespace

StatusOr<std::vector<size_t>> OpenNpy(absl::string_view filename,
                                      absl::string_view expected_descr,
                                      size_t word_size, std::ifstream& in) {
  const std::string fname(filename);
  in.open(fname, std::ifstream::binary);
  if (!in.is_open())
    return NotFoundError(absl::StrCat("Failed to open file ", fname));
  return ReadNpyHeader(filename, expected_descr, word_size, in);
}

StatusOr<std::vector<size_t>> ReadNpyHeader(absl::string_view name,
                                            absl::string_view expected_descr,
                                            size_t word_size,
                                            std::istream& in) {
  const std::string fname(name);
  const auto bad = [&fname](absl::string_view why) {
    return InvalidArgumentError(
        absl::StrCat("Invalid .npy file ", fname, ": ", why));
  };
  in.seekg(0, std::ios::end);
  const std::streamoff file_size_signed = in.tellg();
  in.seekg(0, std::ios::beg);
  if (!in || file_size_signed < 0)
    return InternalError(absl::StrCat("I/O error reading ", fname));
  const uint64_t file_size = static_cast<uint64_t>(file_size_signed);

  unsigned char prefix[12];
  if (file_size < 10 || !in.read(reinterpret_cast<char*>(prefix), 10))
    return bad("too short");
  if (std::memcmp(prefix, "\x93NUMPY", 6) != 0) return bad("bad magic string");
  const int major = prefix[6];
  uint64_t header_len;
  size_t prefix_len;
  if (major == 1) {
    header_len = prefix[8] | (uint64_t{prefix[9]} << 8);
    prefix_len = 10;
  } else if (major == 2 || major == 3) {
    if (file_size < 12 || !in.read(reinterpret_cast<char*>(prefix) + 10, 2))
      return bad("too short");
    header_len = prefix[8] | (uint64_t{prefix[9]} << 8) |
                 (uint64_t{prefix[10]} << 16) | (uint64_t{prefix[11]} << 24);
    prefix_len = 12;
  } else {
    return bad(absl::StrCat("unsupported format version ", major));
  }
  if (header_len > kMaxHeaderLen || header_len > file_size - prefix_len)
    return bad(absl::StrCat("header length ", header_len,
                            " exceeds the file size"));
  std::string header(header_len, '\0');
  if (header_len > 0 && !in.read(header.data(), header_len))
    return InternalError(absl::StrCat("I/O error reading ", fname));
  if (header.empty() || header.back() != '\n')
    return bad("header isn't terminated by a newline");

  std::string descr;
  bool have_descr = false, have_order = false, have_shape = false;
  bool fortran_order = false;
  std::vector<size_t> shape;
  HeaderParser p(header);
  if (!p.Consume('{')) return bad("header isn't a dict");
  while (!p.Consume('}')) {
    std::string key;
    if (!p.ParseString(&key) || !p.Consume(':'))
      return bad("malformed header dict");
    if (key == "descr" && !have_descr) {
      have_descr = true;
      if (!p.ParseString(&descr))
        return bad("unsupported 'descr' (structured dtypes aren't supported)");
    } else if (key == "fortran_order" && !have_order) {
      have_order = true;
      if (p.ConsumeWord("True")) {
        fortran_order = true;
      } else if (!p.ConsumeWord("False")) {
        return bad("malformed 'fortran_order'");
      }
    } else if (key == "shape" && !have_shape) {
      have_shape = true;
      if (!p.Consume('(')) return bad("malformed 'shape'");
      while (!p.Consume(')')) {
        uint64_t dim;
        if (!p.ParseUint(&dim) || dim > std::numeric_limits<size_t>::max())
          return bad("malformed or out-of-range 'shape'");
        shape.push_back(dim);
        if (!p.Consume(',')) {
          if (!p.Consume(')')) return bad("malformed 'shape'");
          break;
        }
      }
    } else {
      return bad(absl::StrCat("unexpected or repeated header key '", key,
                              "'"));
    }
    if (!p.Consume(',')) {
      if (!p.Consume('}')) return bad("malformed header dict");
      break;
    }
  }
  if (!p.AtEnd()) return bad("trailing characters after the header dict");
  if (!have_descr || !have_order || !have_shape)
    return bad("header lacks 'descr', 'fortran_order' or 'shape'");

  absl::string_view want = expected_descr;
  if (want.size() >= 2 && want.front() == '\'' && want.back() == '\'')
    want = want.substr(1, want.size() - 2);
  absl::string_view got = descr;
  if (got.empty() || want.empty() ||
      got.substr(1) != want.substr(1))
    return bad(absl::StrCat("dtype is '", descr, "', expected '", want, "'"));
  const char order = got.front();
  bool order_ok;
  if (word_size == 1) {
    order_ok = order == '<' || order == '>' || order == '|' || order == '=';
  } else {
    order_ok = order == '=' || (order == '<' && HostIsLittleEndian()) ||
               (order == '>' && !HostIsLittleEndian());
  }
  if (!order_ok)
    return bad(absl::StrCat("dtype '", descr,
                            "' has the wrong byte order for this machine"));

  size_t dims_above_one = 0;
  bool any_zero = false;
  for (size_t d : shape) {
    dims_above_one += d > 1;
    any_zero |= d == 0;
  }
  if (fortran_order && dims_above_one > 1)
    return bad("Fortran-order arrays aren't supported");

  uint64_t n_elements = 1;
  if (any_zero) {
    n_elements = 0;
  } else {
    for (size_t d : shape) {
      if (n_elements > std::numeric_limits<uint64_t>::max() / d)
        return bad("shape overflows");
      n_elements *= d;
    }
  }
  if (n_elements > std::numeric_limits<uint64_t>::max() / word_size)
    return bad("shape overflows");
  const uint64_t data_bytes = n_elements * word_size;
  const uint64_t available = file_size - prefix_len - header_len;
  if (data_bytes > available)
    return bad(absl::StrCat("truncated: the shape needs ", data_bytes,
                            " bytes of data, the file has ", available));
  if (data_bytes < available)
    return bad(absl::StrCat("the shape needs ", data_bytes,
                            " bytes of data, but the file has ", available));
  return shape;
}

}  // namespace npy_internal

template <typename T>
struct assert_false : std::false_type {};

template <typename T>
std::string numpy_type_name() {
  static_assert(assert_false<T>::value, "Type incompatible with numpy");
}
template <>
std::string numpy_type_name<uint8_t>() {
  return "'<u1'";
}
template <>
std::string numpy_type_name<uint16_t>() {
  return "'<u2'";
}
template <>
std::string numpy_type_name<uint32_t>() {
  return "'<u4'";
}
template <>
std::string numpy_type_name<uint64_t>() {
  return "'<u8'";
}
template <>
std::string numpy_type_name<int8_t>() {
  return "'<i1'";
}
template <>
std::string numpy_type_name<int16_t>() {
  return "'<i2'";
}
template <>
std::string numpy_type_name<int32_t>() {
  return "'<i4'";
}
template <>
std::string numpy_type_name<int64_t>() {
  return "'<i8'";
}
template <>
std::string numpy_type_name<float>() {
  return "'<f4'";
}
template <>
std::string numpy_type_name<double>() {
  return "'<f8'";
}

}  // namespace research_scann
