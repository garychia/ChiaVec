add_executable(cpp_example main.cpp)
target_include_directories(cpp_example PUBLIC ${PROJECT_SOURCE_DIR}/src)
target_link_libraries(cpp_example chiavec)
