# Simple line detection application

This application will do a simple line detection on image or video.

## Prerequisites

`C++17`
`OpenCV 4.2.0`
`yaml-cpp 0.6.3`

## Build
```
mkdir build && cd build
cmake ../
make -j16
```

## Run
`./drivedetect -i <input_image_or_video> -c <detection_config>`

### Detection config

Detection configuration files can be found under `configs` folder. One example:
```
# Mask coordinates depend on camera position
mask_points:
    - [0, 540]
    - [480, 270]
    - [960, 540]

# Canny positions
canny:
  low_threshold: 100
  high_threshold: 200

# HoughLinesP parameters
hough_lines:
  rho: 5
  theta: 0.0523333
  threshold: 160
  min_line_length: 40
  max_line_gap: 25

# GaussianBlur size
blur_size:
  height: 1
  width: 1

# Slope intercept used for drawing the lines
line_slope: 0.6
```