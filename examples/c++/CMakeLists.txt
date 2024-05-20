add_executable(cpp_example main.cu)
target_include_directories(cpp_example PUBLIC ${PROJECT_SOURCE_DIR}/src)
target_link_libraries(cpp_example cudavec)
