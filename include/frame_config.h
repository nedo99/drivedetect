#ifndef FRAME_CONFIG_H
#define FRAME_CONFIG_H
#include <iostream>
#include <opencv2/core.hpp>

#define YAML_POINTS      "mask_points"
#define YAML_CANNY       "canny"
#define YAML_HOUGH_LINES "hough_lines"
#define YAML_BLUR_SIZE   "blur_size"
#define YAML_SLOPE       "line_slope"

using namespace std;
using namespace cv;

class FrameConfig {
    public:
        FrameConfig();
        FrameConfig(string cfgPath);

        void printConfig() const;
        bool parseConfig();

        vector<Point> getMaskPoints() const {return maskPts;}
        double getRho() const {return rho;}
        double getTheta() const {return theta;}
        int getHlpThreshold() const {return hlpThreshold;}
        double getMinLineLength() const {return minLineLength;}
        double getMaxLineGap() const {return maxLineGap;}
        double getCannyLThr() const {return cannyLowThreshold;}
        double getCannyHThr() const {return cannyHighThreshold;}
        int getBlurHeight() const {return blurHeight;}
        int getBlurWidth() const {return blurWidth;}
        double getSlopeIntercept() const {return slopeIntercept;}
    private:
        string configPath;
        vector<Point> maskPts;
        double rho;
        double theta;
        int hlpThreshold;
        double minLineLength;
        double maxLineGap;
        double cannyLowThreshold;
        double cannyHighThreshold;
        int blurWidth;
        int blurHeight;
        double slopeIntercept;
};

#endif