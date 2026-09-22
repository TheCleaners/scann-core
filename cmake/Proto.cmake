# protoc invocations for scann-core's .proto files.
#
# All of scann-core's protos import each other by workspace-root-relative
# path (e.g. `import "scann/proto/hash.proto";`), and some import protobuf's
# well-known types (`google/protobuf/timestamp.proto`), so both roots go on
# the import path and every file is generated in one protoc invocation.
# SCANN_PROTOBUF_WKT_DIR (set in Dependencies.cmake) is the directory
# containing google/protobuf/*.proto for whichever protobuf is in use.

function(_scann_proto_outputs out_var proto_root out_dir suffix)
  set(outs "")
  foreach(proto ${ARGN})
    file(RELATIVE_PATH rel "${proto_root}" "${proto}")
    string(REGEX REPLACE "\\.proto$" "${suffix}" rel_out "${rel}")
    list(APPEND outs "${out_dir}/${rel_out}")
  endforeach()
  set(${out_var} ${outs} PARENT_SCOPE)
endfunction()

# scann_generate_cpp_protos(<srcs var> <hdrs var>
#                           PROTO_ROOT <dir> OUT_DIR <dir> PROTOS <files...>)
function(scann_generate_cpp_protos srcs_var hdrs_var)
  cmake_parse_arguments(ARG "" "PROTO_ROOT;OUT_DIR" "PROTOS" ${ARGN})
  file(MAKE_DIRECTORY "${ARG_OUT_DIR}")
  _scann_proto_outputs(srcs "${ARG_PROTO_ROOT}" "${ARG_OUT_DIR}" ".pb.cc" ${ARG_PROTOS})
  _scann_proto_outputs(hdrs "${ARG_PROTO_ROOT}" "${ARG_OUT_DIR}" ".pb.h" ${ARG_PROTOS})
  list(LENGTH ARG_PROTOS n)
  add_custom_command(
    OUTPUT ${srcs} ${hdrs}
    COMMAND protobuf::protoc
            "--proto_path=${ARG_PROTO_ROOT}"
            "--proto_path=${SCANN_PROTOBUF_WKT_DIR}"
            "--cpp_out=${ARG_OUT_DIR}"
            ${ARG_PROTOS}
    DEPENDS ${ARG_PROTOS} protobuf::protoc
    COMMENT "Generating C++ protobuf code for scann-core (${n} .proto files)"
    VERBATIM)
  set(${srcs_var} ${srcs} PARENT_SCOPE)
  set(${hdrs_var} ${hdrs} PARENT_SCOPE)
endfunction()

# scann_generate_python_protos(<outputs var>
#                              PROTO_ROOT <dir> OUT_DIR <dir> PROTOS <files...>)
function(scann_generate_python_protos outs_var)
  cmake_parse_arguments(ARG "" "PROTO_ROOT;OUT_DIR" "PROTOS" ${ARGN})
  file(MAKE_DIRECTORY "${ARG_OUT_DIR}")
  _scann_proto_outputs(outs "${ARG_PROTO_ROOT}" "${ARG_OUT_DIR}" "_pb2.py" ${ARG_PROTOS})
  list(LENGTH ARG_PROTOS n)
  add_custom_command(
    OUTPUT ${outs}
    COMMAND protobuf::protoc
            "--proto_path=${ARG_PROTO_ROOT}"
            "--proto_path=${SCANN_PROTOBUF_WKT_DIR}"
            "--python_out=${ARG_OUT_DIR}"
            ${ARG_PROTOS}
    DEPENDS ${ARG_PROTOS} protobuf::protoc
    COMMENT "Generating Python protobuf modules for scann-core (${n} .proto files)"
    VERBATIM)
  set(${outs_var} ${outs} PARENT_SCOPE)
endfunction()
