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

# Line detection related config
line_detection:
  # Mask coordinates depend on camera position
  mask_points:
      - [200, 720]
      - [670, 360]
      - [1110, 720]

  # Canny positions
  canny:
    low_threshold: 50
    high_threshold: 100

  # HoughLinesP parameters
  hough_lines:
    rho: 2
    theta: 0.0174444
    threshold: 100
    min_line_length: 20
    max_line_gap: 150

  # GaussianBlur size
  blur_size:
    height: 5
    width: 5

  # Slope intercept used for drawing the lines
  line_slope: 0.7

# Object detection related config
object_detection:
  model: "ssd_mobilenet_v2_coco_2018_03_29.pb"
  config: "ssd_mobilenet_v2_coco_2018_03_29.pbtxt"
  mean: [0, 0, 0]
  scale: 1.0
  width: 300
  height: 300
  rgb: true
  classes: "object_detection_classes_coco.txt"
  # Confidence threshold
  cnf_threshold: 0.5
  # Non-maximum suppression threshold
  nms_threshold: 0.4
```

### Models

Models are based on OpenCV models. Model and classes used for development can be downloaded from this link: <https://drive.google.com/open?id=1_ZfaLfz48-zjQWfz4yYz46uUQEOjHgQX>