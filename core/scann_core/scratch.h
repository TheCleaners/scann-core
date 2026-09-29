// Copyright 2026 Elias Benali (@ebenali) and TheCleaners.
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

// Per-thread scratch objects for the search path, so that a query doesn't
// allocate and free the same buffers every time (the distances to the
// centroids, top-N buffers, the list of leaves to search, ...).
//
//   ScratchLease<std::vector<float>> distances;
//   distances->resize(n);  // no allocation once the thread has seen n
//
// A lease takes an object from the calling thread's pool (or makes one) and
// puts it back when it goes out of scope, so nested or recursive users on
// the same thread each get their own object, and nothing is shared between
// threads. The object keeps its contents and capacity from its last use:
// users must (re)initialize what they read. Objects that grew beyond
// kMaxRetainedBytes (e.g. from a batched or training-time call) are freed
// instead of kept, and a thread keeps at most kMaxPooled of each type.

#ifndef SCANN_CORE_SCRATCH_H_
#define SCANN_CORE_SCRATCH_H_

#include <cstddef>
#include <memory>
#include <utility>
#include <vector>

namespace scann_core {

inline constexpr size_t kMaxRetainedBytes = size_t{1} << 20;
inline constexpr size_t kMaxPooled = 8;

// Whether a used scratch object is worth keeping. Overload for types whose
// footprint can grow.
template <typename T>
bool ScratchIsRetainable(const T&) {
  return true;
}
template <typename U, typename A>
bool ScratchIsRetainable(const std::vector<U, A>& v) {
  return v.capacity() * sizeof(U) <= kMaxRetainedBytes;
}

template <typename T>
class ScratchLease {
 public:
  ScratchLease() {
    auto& pool = Pool();
    if (pool.empty()) {
      object_ = std::make_unique<T>();
    } else {
      object_ = std::move(pool.back());
      pool.pop_back();
    }
  }
  ~ScratchLease() {
    auto& pool = Pool();
    if (pool.size() < kMaxPooled && ScratchIsRetainable(*object_)) {
      pool.push_back(std::move(object_));
    }
  }
  ScratchLease(const ScratchLease&) = delete;
  ScratchLease& operator=(const ScratchLease&) = delete;

  T& operator*() const { return *object_; }
  T* operator->() const { return object_.get(); }
  T* get() const { return object_.get(); }

 private:
  static std::vector<std::unique_ptr<T>>& Pool() {
    thread_local std::vector<std::unique_ptr<T>> pool;
    return pool;
  }

  std::unique_ptr<T> object_;
};

}  // namespace scann_core

#endif  // SCANN_CORE_SCRATCH_H_
