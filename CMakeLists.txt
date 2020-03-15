cmake_minimum_required(VERSION 3.5)
project(drivedetect)

set(CMAKE_CXX_STANDARD 17)
include_directories(include)
find_package(OpenCV REQUIRED)
find_package(yaml-cpp REQUIRED)

add_subdirectory(src)

add_executable(drivedetect main.cpp)

target_link_libraries(drivedetect frame_parse)