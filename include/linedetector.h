#ifndef LINEDETECTOR_H
#define LINEDETECTOR_H
#include <iostream>

#include <opencv2/core.hpp>

#include "frame_config.h"

using namespace std;
using namespace cv;

class LineDetector {
    public:
        LineDetector(const FrameConfig &config);

        vector<Vec4i> detectLines(const Mat &frame);
        bool init();
    private:
        // Methods
        void polyfit(const Mat& src_x, const Mat& src_y, Mat& dst, int order);
        Mat regionOfInterests(const Mat given_image, const vector<Point> pts);
        vector<Vec4i> averageSlopeIntercept(int height, vector<Vec4i> lines, double slope_intercept);
        Vec4i makeCoordinates(int height, Vec2f line_parameters, double slope_intercept);

        // attributes
        FrameConfig *cfg;
};
#endif