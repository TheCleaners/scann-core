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

// scann_core::AvailableCPUs(): the cgroup quota parser, the SCANN_NUM_THREADS
// override, and (Linux) that the count follows the affinity mask. The thread
// counts of real searchers are checked by tests/python/test_threads.py.

#include <cstdio>
#include <cstdlib>
#include <optional>
#include <string>
#include <thread>

#if defined(__linux__)
#include <sched.h>
#endif

#include "scann_core/available_cpus.h"

namespace {

int failures = 0;

void Expect(bool ok, const std::string& what) {
  if (!ok) {
    std::fprintf(stderr, "FAILED: %s\n", what.c_str());
    ++failures;
  }
}

std::string Str(std::optional<int> v) {
  return v ? std::to_string(*v) : std::string("nullopt");
}

void ExpectQuota(const char* cpu_max, std::optional<int> want) {
  const auto got = scann_core::internal::ParseCgroupV2CpuMax(cpu_max);
  Expect(got == want, std::string("ParseCgroupV2CpuMax(\"") + cpu_max +
                          "\") = " + Str(got) + ", want " + Str(want));
}

}  // namespace

int main() {
  ExpectQuota("max 100000\n", std::nullopt);
  ExpectQuota("max", std::nullopt);
  ExpectQuota("", std::nullopt);
  ExpectQuota("400000 100000\n", 4);
  ExpectQuota("150000 100000", 2);  // 1.5 CPUs: rounded up
  ExpectQuota("50000 100000", 1);
  ExpectQuota("1 100000", 1);
  ExpectQuota("200000", 2);  // period defaults to 100000
  ExpectQuota("-1 100000", std::nullopt);  // cgroup v1's "unlimited"
  ExpectQuota("garbage 100000", std::nullopt);
  ExpectQuota("100000 0", std::nullopt);

  unsetenv("SCANN_NUM_THREADS");
  const int base = scann_core::AvailableCPUs();
  Expect(base >= 1, "AvailableCPUs() >= 1");
  Expect(base <= static_cast<int>(std::thread::hardware_concurrency()) ||
             std::thread::hardware_concurrency() == 0,
         "AvailableCPUs() <= hardware_concurrency()");

  setenv("SCANN_NUM_THREADS", "7", 1);
  Expect(scann_core::AvailableCPUs() == 7, "SCANN_NUM_THREADS=7");
  setenv("SCANN_NUM_THREADS", " 3 ", 1);
  Expect(scann_core::AvailableCPUs() == 3, "SCANN_NUM_THREADS=' 3 '");
  for (const char* bad : {"0", "-2", "x", "", "2.5"}) {
    setenv("SCANN_NUM_THREADS", bad, 1);
    Expect(scann_core::AvailableCPUs() == base,
           std::string("SCANN_NUM_THREADS='") + bad + "' is ignored");
  }
  unsetenv("SCANN_NUM_THREADS");

#if defined(__linux__)
  // Evaluated on every call: restricting the affinity to one CPU is seen at
  // once.
  cpu_set_t original;
  if (sched_getaffinity(0, sizeof(original), &original) == 0 &&
      CPU_COUNT(&original) > 1) {
    Expect(base <= CPU_COUNT(&original), "AvailableCPUs() <= affinity");
    cpu_set_t one;
    CPU_ZERO(&one);
    for (int c = 0; c < CPU_SETSIZE; ++c)
      if (CPU_ISSET(c, &original)) {
        CPU_SET(c, &one);
        break;
      }
    if (sched_setaffinity(0, sizeof(one), &one) == 0) {
      Expect(scann_core::AvailableCPUs() == 1, "AvailableCPUs() follows the affinity");
      sched_setaffinity(0, sizeof(original), &original);
    }
  }
#endif

  if (failures) {
    std::fprintf(stderr, "%d failure(s)\n", failures);
    return 1;
  }
  std::printf("OK\n");
  return 0;
}
