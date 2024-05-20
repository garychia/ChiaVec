add_executable(c_example main.c)
target_include_directories(c_example PUBLIC ${PROJECT_SOURCE_DIR}/src)
target_link_libraries(c_example cudavec)
