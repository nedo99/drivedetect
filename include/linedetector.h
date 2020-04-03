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
    bool advancedLineDetection(Mat &frame);
    bool init();
    double getLeftCurvature() const {return leftCurvature;}
    double getRightCurvature() const {return rightCurvature;}
private:
    // Methods
    vector<Vec4i> averageSlopeIntercept(vector<Vec4i> lines);
    Vec4i makeCoordinates(Vec2f line_parameters);
    void initFrame(const Mat &frame);
    void combBinaryThresh(const Mat &frame, const Mat &gryFrame);
    void computeWhiteYellowBinary(const Mat &frame, const Mat &gryFrame);
    void absXYSobel(const Mat &gryFrame);
    void combineSobels();
    
    // attributes
    FrameConfig *cfg;
    Size frameSize, blur_size;
    Scalar whiteLowerBound, whiteUpperBound, yellowLowerBound, yellowUpperBound;
    Mat clrMask, whiteYellowMask, shapeMsk, gammaArray, sobelXMask, sobelYMask, sobelXYMask,
    sobelX, sobelY, combinedMask, combinedBinary;
    bool frameInitialized;
    int frameY1, frameY2;
    double leftCurvature, rightCurvature;
};
#endif
