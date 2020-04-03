#include "frame_config.h"

#include <fstream>

namespace YAML {
template<>
struct convert<Point> {

  static bool decode(const Node& node, Point& rhs) {
    if(!node.IsSequence() || node.size() != 2) {
      return false;
    }

    rhs.x = (int)node[0].as<int>();
    rhs.y = (int)node[1].as<int>();
    return true;
  }
};
template<>
struct convert<Point2f> {

  static bool decode(const Node& node, Point2f& rhs) {
    if(!node.IsSequence() || node.size() != 2) {
      return false;
    }

    rhs.x = (int)node[0].as<float>();
    rhs.y = (int)node[1].as<float>();
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

AdvancedFrameConfig AdvancedFrameConfig::operator=(const AdvancedFrameConfig &other) {
    chessX = other.chessX;
    chessY = other.chessY;
    xThreshold = other.xThreshold;
    yThreshold = other.yThreshold;
    xyThreshold = other.xyThreshold;
    calibrationPath = other.calibrationPath;
    srcPts = other.srcPts;
    dstPts = other.dstPts;
    combKSize = other.combKSize;
    angleThreshold = other.angleThreshold;
    margin = other.margin;
    nSegments = other.nSegments;
    xmPerPix = other.xmPerPix;
    ymPerPix = other.ymPerPix;
    return *this;
}

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
    additionalImageProcessing = cfg.additionalImageProcessing;
    advCfg = cfg.advCfg;
}

bool FrameConfig::parseConfig() {
    YAML::Node config = YAML::LoadFile(configPath);

    if (!config[YAML_LINE_DETECT]) {
      cerr << YAML_LINE_DETECT << " missing in yaml config!" << endl;
      return false;
    }

    if (!config[YAML_OBJ_DETECT]) {
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

        // Check for calibration part
        if (config["advanced_image_processing"]) {
            additionalImageProcessing = true;
            YAML::Node tmpConfig = config["advanced_image_processing"];
            advCfg.scale = tmpConfig["scale"].as<float>();
            if (advCfg.scale > 1) {
                cerr << "Scale value should not be > 1!" << endl;
                return false;
            }
            advCfg.calibrationPath = tmpConfig["calibration_path"].as<string>();
            advCfg.chessX = tmpConfig["chess_x"].as<int>();
            advCfg.chessY = tmpConfig["chess_y"].as<int>();
            advCfg.margin = (int)(tmpConfig["margin"].as<int>() * advCfg.scale);
            advCfg.nSegments = tmpConfig["segments_number"].as<int>();
            advCfg.xmPerPix = tmpConfig["xm_per_pix"].as<double>() * advCfg.scale;
            advCfg.ymPerPix = tmpConfig["ym_per_pix"].as<double>() * advCfg.scale;
            advCfg.combKSize = tmpConfig["sobel"]["comb_ksize"].as<int>();
            advCfg.xThreshold = tmpConfig["sobel"]["x_threshold"].as<Scalar>();
            advCfg.yThreshold = tmpConfig["sobel"]["y_threshold"].as<Scalar>();
            advCfg.xyThreshold = tmpConfig["sobel"]["xy_threshold"].as<Scalar>();
            advCfg.srcPts = tmpConfig["src_points"].as<vector<Point2f>>();
            for (unsigned long i = 0; i < advCfg.srcPts.size(); i++) {
                advCfg.srcPts[i].x = advCfg.srcPts[i].x * advCfg.scale;
                advCfg.srcPts[i].y = advCfg.srcPts[i].y * advCfg.scale;
            }
            advCfg.dstPts = tmpConfig["dst_points"].as<vector<Point2f>>();
            for (unsigned long i = 0; i < advCfg.dstPts.size(); i++) {
                advCfg.dstPts[i].x = advCfg.dstPts[i].x * advCfg.scale;
                advCfg.dstPts[i].y = advCfg.dstPts[i].y * advCfg.scale;
            }
            double lowAngleThresh, upperAngleThresh;
            lowAngleThresh = tmpConfig["sobel"]["lower_angle_threshold"].as<double>();
            upperAngleThresh = tmpConfig["sobel"]["upper_angle_threshold"].as<double>();
            advCfg.angleThreshold = Scalar(lowAngleThresh, upperAngleThresh);
        }
    }
    catch (YAML::BadConversion &e) {
        cerr << "Missing key or wrong attribute in yaml config!" << endl << e.what() << endl;
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
