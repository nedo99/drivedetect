#include "frame_config.h"

#include "yaml-cpp/yaml.h"

namespace YAML {
template<>
struct convert<Point> {
  static Node encode(const Point& rhs) {
    Node node;
    node.push_back(rhs.x);
    node.push_back(rhs.y);
    return node;
  }

  static bool decode(const Node& node, Point& rhs) {
    if(!node.IsSequence() || node.size() != 2) {
      return false;
    }

    rhs.x = node[0].as<int>();
    rhs.y = node[1].as<int>();
    return true;
  }
};
}

static vector<string> yaml_expected_keys = {YAML_POINTS, YAML_CANNY, YAML_HOUGH_LINES, YAML_BLUR_SIZE, YAML_SLOPE};

FrameConfig::FrameConfig(string cfgPath) {
   configPath = cfgPath;
}

bool FrameConfig::parseConfig() {
    YAML::Node config = YAML::LoadFile(configPath);
    for (int i = 0; i < yaml_expected_keys.size(); i ++) {
        if (config[yaml_expected_keys[i]].IsNull()) {
            cout << yaml_expected_keys[i] << " parameter missing in yaml config!" << endl;
            return false;
        }
    }
    
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
    }
    catch (YAML::BadConversion &e) {
        cout << "Missing key or wrong attribute in yaml config!" << endl;
        return false;
    }
    return true;
}

void FrameConfig::printConfig() const {
    cout << "-------------------------" << endl;
    cout << "Mask points number: " << maskPts.size() << endl;
    for (int i = 0; i < maskPts.size(); i++)
        cout << "[" << maskPts.at(i).x << ", " << maskPts.at(i).y << "] ";
    cout << endl;
    cout << "Canny low threshold: " << cannyLowThreshold << "; Canny high threshold " << cannyHighThreshold << endl;
    cout << "HoughLinesP rho " << rho << "; theta " << theta << "; threshold " << hlpThreshold;
    cout << "; min line length " << minLineLength << "; max line gap " << maxLineGap << endl;
    cout << "-------------------------" << endl;
}