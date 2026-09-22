# cmake -DAR=<ar> -DSCRIPT=<mri script> -DOUTPUT=<archive> -P run_ar_mri.cmake
# Runs `ar -M` with an MRI script on stdin (add_custom_command can't do shell
# redirection portably).
file(REMOVE "${OUTPUT}")
execute_process(COMMAND "${AR}" -M INPUT_FILE "${SCRIPT}" RESULT_VARIABLE rc)
if(NOT rc EQUAL 0)
  message(FATAL_ERROR "ar -M failed (${rc}) for ${SCRIPT}")
endif()
