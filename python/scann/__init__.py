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

"""scann-core's `scann` package: upstream's API without the TensorFlow op."""

from scann.scann_ops.py.scann_builder import ReorderType
from scann.scann_ops.py.scann_builder import ScannBuilder
from scann.scann_ops.py import scann_ops_pybind
from scann.version import __version__

__all__ = ["ReorderType", "ScannBuilder", "scann_ops_pybind", "__version__"]
