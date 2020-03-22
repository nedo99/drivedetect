#include "frame_config.h"

#include <fstream>

namespace YAML {
template<>
struct convert<Point> {

  static bool decode(const Node& node, Point& rhs) {
    if(!node.IsSequence() || node.size() != 2) {
      return false;
    }

    rhs.x = node[0].as<int>();
    rhs.y = node[1].as<int>();
    return true;
  }
};
template<>
struct convert<Scalar> {

  static bool decode(const Node& node, Scalar& rhs) {
    if(!node.IsSequence() || node.size() < 3) {
      return false;
    }
    for (uint32_t i = 0; i < node.size(); i++) {
      rhs[i] = node[i].as<int>();
    }
    return true;
  }
};
}

static vector<string> yaml_expected_keys = {YAML_POINTS, YAML_CANNY, YAML_HOUGH_LINES, YAML_BLUR_SIZE, YAML_SLOPE, YAML_OBJ_DETECT};

FrameConfig::FrameConfig(string cfgPath) {
   configPath = cfgPath;
}

FrameConfig::FrameConfig(const FrameConfig &cfg) {
    configPath = cfg.configPath;
    maskPts = cfg.maskPts;
    classes = cfg.classes;
    rho = cfg.rho;
    theta = cfg.theta;
    hlpThreshold = cfg.hlpThreshold;
    minLineLength = cfg.minLineLength;
    maxLineGap = cfg.maxLineGap;
    cannyLowThreshold = cfg.cannyLowThreshold;
    cannyHighThreshold = cfg.cannyHighThreshold;
    blurWidth = cfg.blurWidth;
    blurHeight = cfg.blurHeight;
    slopeIntercept = cfg.slopeIntercept;
    dnn_model = cfg.dnn_model;
    dnn_config = cfg.dnn_config;
    mean = cfg.mean;
    scale = cfg.scale;
    inpSize = cfg.inpSize;
    swapRb = cfg.swapRb;
    confThreshold = cfg.confThreshold;
    nmsThreshold = cfg.nmsThreshold;
    gammaConf = cfg.gammaConf;
}

bool FrameConfig::parseConfig() {
    YAML::Node config = YAML::LoadFile(configPath);
    for (uint64_t i = 0; i < yaml_expected_keys.size(); i ++) {
        if (config[yaml_expected_keys[i]].IsNull()) {
            cout << yaml_expected_keys[i] << " parameter missing in yaml config!" << endl;
            return false;
        }
    }

    if (config[YAML_LINE_DETECT].IsNull()) {
      cerr << YAML_LINE_DETECT << " missing in yaml config!" << endl;
      return false;
    }

    if (config[YAML_OBJ_DETECT].IsNull()) {
      cerr << YAML_OBJ_DETECT << " missing in yaml config!" << endl;
      return false;
    }

    if (!parseLineDetectionConfig(config[YAML_LINE_DETECT])) {
      return false;
    }

    if (!parseObjectDetectionConfig(config[YAML_OBJ_DETECT])) {
      return false;
    }

    return true;
}

bool FrameConfig::parseLineDetectionConfig(YAML::Node config) {
  try {
        maskPts = config[YAML_POINTS].as<vector<Point>>();
        cannyLowThreshold = config[YAML_CANNY]["low_threshold"].as<int>();
        cannyHighThreshold = config[YAML_CANNY]["high_threshold"].as<int>();
        rho = config[YAML_HOUGH_LINES]["rho"].as<int>();
        theta = config[YAML_HOUGH_LINES]["theta"].as<double>();
        hlpThreshold = config[YAML_HOUGH_LINES]["threshold"].as<int>();
        minLineLength = config[YAML_HOUGH_LINES]["min_line_length"].as<int>();
        maxLineGap = config[YAML_HOUGH_LINES]["max_line_gap"].as<int>();
        blurHeight = config[YAML_BLUR_SIZE]["height"].as<int>();
        blurWidth = config[YAML_BLUR_SIZE]["width"].as<int>();
        slopeIntercept = config[YAML_SLOPE].as<double>();
        gammaConf = config["gamma_cof"].as<float>();
    }
    catch (YAML::BadConversion &e) {
        cerr << "Missing key or wrong attribute in yaml config!" << endl;
        return false;
    }

    return true;
}

bool FrameConfig::parseObjectDetectionConfig(YAML::Node config) {
  try {
    mean = config["mean"].as<Scalar>();
    scale = config["scale"].as<float>();
    int width = config["width"].as<int>();
    int height = config["height"].as<int>();
    inpSize = Size(width, height);
    swapRb = config["rgb"].as<bool>();
    string classesPath = config["classes"].as<string>();
    ifstream ifs(classesPath.c_str());
    if (!ifs.is_open()) {
      cerr << "Could not open classes file " << classesPath << "!" << endl;
      return false;
    }
    string line;
    while (getline(ifs, line))
    {
      classes.push_back(line);
    }
    dnn_model = config["model"].as<string>();
    dnn_config = config["config"].as<string>();
    confThreshold = config["cnf_threshold"].as<float>();
    nmsThreshold = config["nms_threshold"].as<float>();
  }
  catch (YAML::BadConversion &e) {
    cerr << "Missing key or wrong attribute in yaml config!" << endl;
    return false;
  }
  return true;
}

void FrameConfig::printConfig() const {
    cout << "-------------------------" << endl;
    cout << "Mask points number: " << maskPts.size() << endl;
    for (uint64_t i = 0; i < maskPts.size(); i++)
        cout << "[" << maskPts.at(i).x << ", " << maskPts.at(i).y << "] ";
    cout << endl;
    cout << "Canny low threshold: " << cannyLowThreshold << "; Canny high threshold " << cannyHighThreshold << endl;
    cout << "HoughLinesP rho " << rho << "; theta " << theta << "; threshold " << hlpThreshold;
    cout << "; min line length " << minLineLength << "; max line gap " << maxLineGap << endl;
    cout << "-------------------------" << endl;
}