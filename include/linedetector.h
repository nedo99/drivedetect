#ifndef LINEDETECTOR_H
#define LINEDETECTOR_H
#include <iostream>

#include <opencv2/core.hpp>

#include "frame_config.h"

#define GAMMA_LIMIT 256

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
        vector<Vec4i> averageSlopeIntercept(vector<Vec4i> lines);
        Vec4i makeCoordinates(Vec2f line_parameters);
        void initFrame(const Mat &frame);

        // attributes
        FrameConfig *cfg;
        Size frameSize, blur_size;
        Scalar whiteLowerBound, whiteUpperBound, yellowLowerBound, yellowUpperBound;
        Mat whiteMask, yellowMask, clrMask, shapeMsk, gammaArray;
        bool frameInitialized;
        int frameY1, frameY2;
};
#endif