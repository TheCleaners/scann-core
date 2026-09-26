# Copyright 2026 ebenali and TheCleaners.
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

# cmake -DAR=<ar> -DSCRIPT=<mri script> -DOUTPUT=<archive> -P run_ar_mri.cmake
# Runs `ar -M` with an MRI script on stdin (add_custom_command can't do shell
# redirection portably).
file(REMOVE "${OUTPUT}")
execute_process(COMMAND "${AR}" -M INPUT_FILE "${SCRIPT}" RESULT_VARIABLE rc)
if(NOT rc EQUAL 0)
  message(FATAL_ERROR "ar -M failed (${rc}) for ${SCRIPT}")
endif()
