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

// How many threads scann-core uses by default: the query pool of every
// searcher (search_batched_parallel, mutations) and training with
// training_threads = 0.

#ifndef SCANN_CORE_AVAILABLE_CPUS_H_
#define SCANN_CORE_AVAILABLE_CPUS_H_

#include <optional>
#include <string_view>

namespace scann_core {

// The number of CPUs this process may run on, evaluated on every call (Ray,
// Spark and job launchers often set the affinity after the process started):
//  1. the SCANN_NUM_THREADS environment variable, if it is a positive
//     integer (it wins over everything below, even above the CPU count);
//  2. otherwise the minimum of
//     - the CPUs in the calling thread's affinity mask (sched_getaffinity,
//       so taskset, numactl, cpusets and container cpusets count; hosts with
//       more than 1024 CPUs work), and
//     - the cgroup CPU quota, rounded up: cgroup v2 cpu.max, the minimum
//       over the process's cgroup and its ancestors, or cgroup v1
//       cpu.cfs_quota_us / cpu.cfs_period_us (docker --cpus, Kubernetes CPU
//       limits);
//     like Python's os.process_cpu_count() plus the container limits that
//     the JVM and Go runtimes apply.
// Elsewhere than Linux: the online CPU count. Always at least 1.
int AvailableCPUs();

namespace internal {
// Exposed for tests. The CPU quota in `cpu_max` (a cgroup v2 cpu.max file's
// contents: "max 100000" or "<quota> <period>"), rounded up; nullopt when
// unlimited or unparseable.
std::optional<int> ParseCgroupV2CpuMax(std::string_view cpu_max);
// SCANN_NUM_THREADS's value, if it is a positive integer.
std::optional<int> NumThreadsOverride();
}  // namespace internal

}  // namespace scann_core

#endif  // SCANN_CORE_AVAILABLE_CPUS_H_
