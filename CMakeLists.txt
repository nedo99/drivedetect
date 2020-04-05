cmake_minimum_required(VERSION 3.5)
project(drivedetect)

set(CMAKE_CXX_FLAGS "-Wextra -Werror -Wall")
set(CMAKE_CXX_STANDARD 17)

set(VALIDATE_PARSING 1)
#add_compile_definitions(VALIDATE_PARSING)

include_directories(include)

find_package(OpenCV REQUIRED)
find_package(yaml-cpp REQUIRED)

add_subdirectory(src)

add_executable(drivedetect main.cpp)

if(UNIX AND NOT APPLE)
    # Filesystem library needs to be link on Linux with c++17
    target_link_libraries(drivedetect frame_parse stdc++fs)
else()
    target_link_libraries(drivedetect frame_parse)
endif()