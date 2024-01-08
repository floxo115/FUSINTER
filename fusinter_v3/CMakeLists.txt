cmake_minimum_required(VERSION 3.25)
project(FUSINTER_v3)

set(CMAKE_CXX_STANDARD 20)

add_subdirectory(external/eigen)
add_subdirectory(external/Catch2)

add_library(FUSINTER_v3 lib/library.cpp paper_data.h lib/TableManager.h lib/Splitter.h lib/typedefs.h lib/errors.h)
target_include_directories(FUSINTER_v3 PUBLIC external/eigen)

add_executable(TEST lib/tests.cpp)
target_include_directories(TEST PUBLIC external/eigen lib)

target_link_libraries(TEST PRIVATE Catch2::Catch2WithMain)
target_include_directories(TEST PUBLIC external/Catch2/src)

add_executable(MAIN main.cpp paper_data.h)
target_link_libraries(MAIN PRIVATE FUSINTER_v3)
