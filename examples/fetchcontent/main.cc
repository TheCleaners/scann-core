// Copyright 2026 ebenali and TheCleaners.
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

// Builds a small index and searches it: enough to show that scann-core
// compiles and links from a separate project.

#include <cstdio>
#include <random>
#include <vector>

#include "scann/scann_ops/cc/scann.h"
#include "scann_core/config_builder.h"
#include "scann_core/version.h"

int main() {
  constexpr size_t n = 2000, dim = 32;
  std::mt19937 rng(0);
  std::normal_distribution<float> gauss;
  std::vector<float> data(n * dim);
  for (float& x : data) x = gauss(rng);

  scann_core::TreeOptions tree;
  tree.num_leaves = 40;
  tree.num_leaves_to_search = 10;
  auto config = scann_core::ConfigBuilder(5, scann_core::DistanceMeasure::kSquaredL2, dim)
                    .Tree(tree)
                    .ScoreAh({})
                    .Reorder({50})
                    .BuildText(n);
  if (!config.ok()) return std::fprintf(stderr, "%s\n", config.status().ToString().c_str()), 1;

  research_scann::ScannInterface index;
  if (auto s = index.Initialize(data, n, *config, 0); !s.ok())
    return std::fprintf(stderr, "%s\n", s.ToString().c_str()), 1;
  research_scann::NNResultsVector result;
  research_scann::DatapointPtr<float> query(nullptr, data.data() + 7 * dim, dim, dim);
  if (auto s = index.Search(query, &result, -1, -1, -1); !s.ok())
    return std::fprintf(stderr, "%s\n", s.ToString().c_str()), 1;
  std::printf("scann-core %s: nearest neighbour of point 7 is %u\n",
              SCANN_CORE_VERSION, result.front().first);
  return result.front().first == 7 ? 0 : 1;
}
