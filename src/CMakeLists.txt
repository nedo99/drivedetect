
add_library(frame_parse frame_parse.cpp frame_config.cpp objectdetector.cpp linedetector.cpp detection_helper.cpp)

target_link_libraries(frame_parse ${OpenCV_LIBS} yaml-cpp)
