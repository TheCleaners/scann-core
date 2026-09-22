add_executable(scann_core_example_quickstart cpp/quickstart.cc)
target_link_libraries(scann_core_example_quickstart PRIVATE scann::core)
set_target_properties(scann_core_example_quickstart PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}")

# The tutorial's C++ program (docs/tutorial/07-cpp-and-rust.md).
add_executable(scann_core_tutorial_glove ../docs/tutorial/code/cpp/glove.cc)
target_link_libraries(scann_core_tutorial_glove PRIVATE scann::core)
set_target_properties(scann_core_tutorial_glove PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}")
