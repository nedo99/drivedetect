cmake_minimum_required(VERSION 3.5)
project(drivedetect)

set(CMAKE_CXX_FLAGS "-Wextra -Werror -Wall -Wno-deprecated-copy")
set(CMAKE_CXX_STANDARD 17)

set(VALIDATE_PARSING 1)
#add_compile_definitions(VALIDATE_PARSING)
add_definitions(-DOPENCV_DISABLE_EIGEN_TENSOR_SUPPORT)

include_directories(include)

find_package(OpenCV REQUIRED)
find_package(yaml-cpp REQUIRED)
find_package(Eigen3 REQUIRED)

add_subdirectory(src)

add_executable(drivedetect main.cpp)

if(UNIX AND NOT APPLE)
    # Filesystem library needs to be link on Linux with c++17
    target_link_libraries(drivedetect frame_parse stdc++fs ${Eigen_LIBRARIES})
else()
    target_link_libraries(drivedetect frame_parse)
endif()