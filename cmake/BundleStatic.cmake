# scann_bundle_static_deps(<target>
#     ROOT <target whose link dependencies to bundle>
#     OUTPUT <path to the merged .a>
#     SYSTEM_LIBS_VAR <variable receiving non-archive link items>)
#
# Walks ROOT's transitive link interface, merges every static library found
# (abseil's ~100 archives, libprotobuf, utf8_range, highway, cnpy, zlib) into
# a single archive, and reports everything else (-pthread, -lm, system
# shared libraries) through SYSTEM_LIBS_VAR. A single archive also removes
# link-order concerns: the linker rescans one archive until it's closed.
#
# This exists for consumers that don't use CMake (the Rust crate's
# build.rs); CMake consumers should just link scann::core_static.

set(_SCANN_BUNDLE_AR_SCRIPT "${CMAKE_CURRENT_LIST_DIR}/run_ar_mri.cmake")

function(scann_bundle_static_deps name)
  cmake_parse_arguments(ARG "" "ROOT;OUTPUT;SYSTEM_LIBS_VAR" "" ${ARGN})

  set(queue ${ARG_ROOT})
  set(seen "")
  set(static_libs "")
  set(system_libs "")
  set(skipped "")
  while(queue)
    list(POP_FRONT queue item)
    if(item MATCHES "^\\$<LINK_ONLY:(.+)>$")
      set(item "${CMAKE_MATCH_1}")
    endif()
    if(item MATCHES "^\\$<BUILD_INTERFACE:(.+)>$")
      set(item "${CMAKE_MATCH_1}")
    endif()
    if(item MATCHES "^\\$<INSTALL_INTERFACE:" OR item STREQUAL "")
      continue()
    endif()
    # Conditions abseil/protobuf actually use, evaluated at configure time:
    #   $<$<BOOL:cond>:value>   and   $<$<PLATFORM_ID:ids>:value>
    if(item MATCHES "^\\$<\\$<BOOL:([^>]*)>:(.*)>$")
      set(cond "${CMAKE_MATCH_1}")
      set(value "${CMAKE_MATCH_2}")
      if(cond AND NOT cond MATCHES "-NOTFOUND$")
        list(APPEND queue "${value}")
      endif()
      continue()
    endif()
    if(item MATCHES "^\\$<\\$<PLATFORM_ID:([^>]*)>:(.*)>$")
      string(REPLACE "," ";" ids "${CMAKE_MATCH_1}")
      set(value "${CMAKE_MATCH_2}")
      if(CMAKE_SYSTEM_NAME IN_LIST ids)
        list(APPEND queue "${value}")
      endif()
      continue()
    endif()
    if(item MATCHES "\\$<")
      list(APPEND skipped "${item}")
      continue()
    endif()
    if(TARGET "${item}")
      get_target_property(aliased "${item}" ALIASED_TARGET)
      if(aliased)
        set(item "${aliased}")
      endif()
    endif()
    if(item IN_LIST seen)
      continue()
    endif()
    list(APPEND seen "${item}")

    if(TARGET "${item}")
      get_target_property(type "${item}" TYPE)
      get_target_property(imported "${item}" IMPORTED)
      if(type STREQUAL "STATIC_LIBRARY")
        list(APPEND static_libs "${item}")
      elseif(imported AND type MATCHES "SHARED_LIBRARY|UNKNOWN_LIBRARY")
        list(APPEND system_libs "$<TARGET_FILE:${item}>")
      endif()
      if(type MATCHES "STATIC_LIBRARY|INTERFACE_LIBRARY|UNKNOWN_LIBRARY|SHARED_LIBRARY")
        get_target_property(deps "${item}" INTERFACE_LINK_LIBRARIES)
        if(deps)
          list(APPEND queue ${deps})
        endif()
      endif()
    elseif(item MATCHES "^-" OR IS_ABSOLUTE "${item}")
      list(APPEND system_libs "${item}")
    elseif(item MATCHES "::")
      # A namespaced name that isn't a target visible here (imported targets
      # are directory-scoped). Turning it into -l<name> would be wrong.
      message(FATAL_ERROR
        "scann-core bundle: link item '${item}' is not a target visible from "
        "${CMAKE_CURRENT_SOURCE_DIR}; import it at the top level (see Dependencies.cmake).")
    else()
      list(APPEND system_libs "-l${item}")
    endif()
  endwhile()

  if(skipped)
    message(STATUS "scann-core bundle: skipped conditional link items: ${skipped}")
  endif()
  list(REMOVE_DUPLICATES system_libs)

  set(script "${CMAKE_CURRENT_BINARY_DIR}/${name}.mri")
  set(content "CREATE ${ARG_OUTPUT}\n")
  foreach(lib ${static_libs})
    string(APPEND content "ADDLIB $<TARGET_FILE:${lib}>\n")
  endforeach()
  string(APPEND content "SAVE\nEND\n")
  file(GENERATE OUTPUT "${script}" CONTENT "${content}")

  set(lib_files "")
  foreach(lib ${static_libs})
    list(APPEND lib_files "$<TARGET_FILE:${lib}>")
  endforeach()

  if(APPLE)
    find_program(SCANN_LIBTOOL libtool REQUIRED)
    add_custom_command(OUTPUT "${ARG_OUTPUT}"
      COMMAND "${SCANN_LIBTOOL}" -static -o "${ARG_OUTPUT}" ${lib_files}
      DEPENDS ${static_libs}
      COMMENT "Bundling ${name} (${ARG_OUTPUT})"
      VERBATIM)
  else()
    add_custom_command(OUTPUT "${ARG_OUTPUT}"
      COMMAND "${CMAKE_COMMAND}" "-DAR=${CMAKE_AR}" "-DSCRIPT=${script}" "-DOUTPUT=${ARG_OUTPUT}"
              -P "${_SCANN_BUNDLE_AR_SCRIPT}"
      DEPENDS ${static_libs} "${script}" "${_SCANN_BUNDLE_AR_SCRIPT}"
      COMMENT "Bundling ${name} (${ARG_OUTPUT})"
      VERBATIM)
  endif()
  add_custom_target(${name} ALL DEPENDS "${ARG_OUTPUT}")

  list(LENGTH static_libs n)
  message(STATUS "scann-core bundle: ${n} static archives -> ${ARG_OUTPUT}")
  set(${ARG_SYSTEM_LIBS_VAR} ${system_libs} PARENT_SCOPE)
endfunction()
