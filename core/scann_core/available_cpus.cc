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

#include "scann_core/available_cpus.h"

#include <algorithm>
#include <cerrno>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <limits>
#include <optional>
#include <sstream>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

#if defined(__linux__)
#include <sched.h>
#endif

#include "absl/strings/numbers.h"
#include "absl/strings/str_split.h"
#include "absl/strings/strip.h"

namespace scann_core {
namespace internal {

std::optional<int> NumThreadsOverride() {
  const char* env = std::getenv("SCANN_NUM_THREADS");
  if (env == nullptr) return std::nullopt;
  int n = 0;
  if (!absl::SimpleAtoi(absl::StripAsciiWhitespace(env), &n) || n < 1)
    return std::nullopt;
  return n;
}

std::optional<int> ParseCgroupV2CpuMax(std::string_view cpu_max) {
  std::vector<std::string_view> parts =
      absl::StrSplit(absl::StripAsciiWhitespace(cpu_max), ' ',
                     absl::SkipWhitespace());
  if (parts.empty() || parts[0] == "max") return std::nullopt;
  int64_t quota = 0, period = 100000;
  if (!absl::SimpleAtoi(parts[0], &quota) || quota <= 0) return std::nullopt;
  if (parts.size() > 1 &&
      (!absl::SimpleAtoi(parts[1], &period) || period <= 0))
    return std::nullopt;
  const int64_t cpus = (quota + period - 1) / period;
  return static_cast<int>(
      std::clamp<int64_t>(cpus, 1, std::numeric_limits<int>::max()));
}

}  // namespace internal

namespace {

#if defined(__linux__)

std::optional<std::string> ReadFile(const std::string& path) {
  std::ifstream f(path);
  if (!f) return std::nullopt;
  std::stringstream ss;
  ss << f.rdbuf();
  return ss.str();
}

std::optional<int> AffinityCPUs() {
  // CPU_ALLOC: a plain cpu_set_t holds 1024 CPUs, and sched_getaffinity
  // fails with EINVAL when the kernel's mask is larger.
  for (int ncpus = 1024; ncpus <= (1 << 22); ncpus *= 2) {
    cpu_set_t* set = CPU_ALLOC(ncpus);
    if (set == nullptr) return std::nullopt;
    const size_t size = CPU_ALLOC_SIZE(ncpus);
    CPU_ZERO_S(size, set);
    if (sched_getaffinity(0, size, set) == 0) {
      const int count = CPU_COUNT_S(size, set);
      CPU_FREE(set);
      if (count > 0) return count;
      return std::nullopt;
    }
    const int err = errno;
    CPU_FREE(set);
    if (err != EINVAL) return std::nullopt;
  }
  return std::nullopt;
}

// The parent of a cgroup path ("/a/b" -> "/a", "/a" -> "/"), or nullopt at
// the root.
std::optional<std::string> Parent(const std::string& path) {
  if (path.empty() || path == "/") return std::nullopt;
  const size_t slash = path.find_last_of('/');
  if (slash == std::string::npos || slash == 0) return std::string("/");
  return path.substr(0, slash);
}

std::string Join(const std::string& root, const std::string& path) {
  return path == "/" ? root : root + path;
}

// The smallest quota over the cgroup and its ancestors. In a container
// without a cgroup namespace, /proc/self/cgroup names the host's path, which
// doesn't exist under the container's /sys/fs/cgroup; walking up then ends
// at the container's own root cgroup, which holds its limit.
std::optional<int> CgroupCPUs() {
  std::optional<std::string> cgroup = ReadFile("/proc/self/cgroup");
  if (!cgroup) return std::nullopt;
  std::optional<int> limit;
  auto take = [&limit](std::optional<int> v) {
    if (v && (!limit || *v < *limit)) limit = v;
  };
  for (std::string_view line : absl::StrSplit(*cgroup, '\n')) {
    // hierarchy-ID:controller-list:path
    std::vector<std::string_view> f = absl::StrSplit(line, absl::MaxSplits(':', 2));
    if (f.size() != 3) continue;
    std::string path(f[2]);
    if (path.empty() || path[0] != '/') continue;
    if (f[0] == "0" && f[1].empty()) {
      // cgroup v2 (unified).
      for (std::optional<std::string> p = path; p; p = Parent(*p)) {
        if (auto v = ReadFile(Join("/sys/fs/cgroup", *p) + "/cpu.max"))
          take(internal::ParseCgroupV2CpuMax(*v));
      }
      continue;
    }
    // cgroup v1: the "cpu" controller (usually mounted with cpuacct).
    bool has_cpu = false;
    for (std::string_view c : absl::StrSplit(f[1], ','))
      has_cpu |= (c == "cpu");
    if (!has_cpu) continue;
    for (const char* mount :
         {"/sys/fs/cgroup/cpu,cpuacct", "/sys/fs/cgroup/cpuacct,cpu",
          "/sys/fs/cgroup/cpu"}) {
      for (std::optional<std::string> p = path; p; p = Parent(*p)) {
        const std::string dir = Join(mount, *p);
        auto quota = ReadFile(dir + "/cpu.cfs_quota_us");
        auto period = ReadFile(dir + "/cpu.cfs_period_us");
        if (quota && period)
          take(internal::ParseCgroupV2CpuMax(
              std::string(absl::StripAsciiWhitespace(*quota)) + " " +
              std::string(absl::StripAsciiWhitespace(*period))));
      }
    }
  }
  return limit;
}

#endif  // __linux__

}  // namespace

int AvailableCPUs() {
  if (auto n = internal::NumThreadsOverride()) return *n;
  int cpus = static_cast<int>(std::thread::hardware_concurrency());
#if defined(__linux__)
  if (auto affinity = AffinityCPUs()) cpus = *affinity;
  if (auto quota = CgroupCPUs()) cpus = std::min(cpus, *quota);
#endif
  return std::max(cpus, 1);
}

}  // namespace scann_core
