//
//  detection_helper.hpp
//  
//
//  Created by Nedim Hadzic
//

#ifndef detection_helper_hpp
#define detection_helper_hpp

#include <stdio.h>
#include <algorithm> /* max_element */
#include <opencv2/core.hpp>

using namespace std;
using namespace cv;

void skipSlideWindow(const Mat &nonZero, Mat &leftFitX, Mat &rightFitX, int margin);
void polyfit(const Mat& src_x, const Mat& src_y, Mat& dst, int order);
void measeureCurvature(const Mat &ploty, const Mat &leftFitX, const Mat &rightFitX, const double &ym_per_pix, const double &xm_per_pix,
                       double &leftCurvature, double &rightCurvature);
Mat linespace(int start, int end, int samples);
Mat getFitX(const Mat &polyfit, const Mat &ploty);
void drawLaneLines(Mat &frame, const Mat &persImg, const Mat &persFrameInv, Mat &leftFitX, Mat &rightFitX,
                   const Mat &ploty);
void perspectiveTransform(const Mat &frame, const vector<Point2f> &srcPts, const vector<Point2f> &dstPts,
                          Mat &outFrame, Mat &persTransformInv);
bool computeLineLanes(const Mat &persImg, const Mat &nonZero, const vector<int> &histogram,
                        Mat &leftFitX, Mat &rightFitX, int margin, int nWindows);
void scaledSobel(const Mat &sobelFrame, Mat &maskFrame, const Scalar &thres);


static inline void updateImg(Mat &outImg, const Mat &y, const Mat &x, const Vec3b &clr) {
    for (int i = 0; i < x.rows; i++) {
        outImg.at<Vec3b>(Point(x.at<int>(i, 0), y.at<int>(i, 0))) = clr;
    }
}

static inline void getFilteredArray(const Mat &nonZero, const Mat &filter, Mat &x, Mat &y) {
    x = Mat(filter.rows, filter.cols, CV_32S);
    y = Mat(filter.rows, filter.cols, CV_32S);
    for (int i = 0; i < filter.rows; i++){
        x.at<int>(i, 0) = nonZero.at<Point>(filter.at<int>(i, 0), 0).x;
        y.at<int>(i, 0) = nonZero.at<Point>(filter.at<int>(i, 0), 0).y;
    }
}

static inline void getGoodIndices(const Mat &nonZero, Mat &goodIndices, float &mean,
                           const Scalar &xThreshold, const Scalar &yThreshold) {
    uint64_t sum = 0;
    for (int i = 0; i < nonZero.rows; i++) {
        if (nonZero.at<Point>(i, 0).x >= xThreshold[0] && nonZero.at<Point>(i, 0).x < xThreshold[1]
            && nonZero.at<Point>(i, 0).y >= yThreshold[0] && nonZero.at<Point>(i, 0).y < yThreshold[1]) {
            goodIndices.push_back(i);
            sum += nonZero.at<Point>(i, 0).x;
        }
    }
    // Just to guard in case there are no elements
    if (goodIndices.rows)
        mean = sum/goodIndices.rows;
}

static inline void getLeftAndRightHistMaximum(const vector<int> histogram, int &leftMax, int &rightMax) {
    int midpoint = (int)histogram.size() / 2;
    vector<int> leftSide = {histogram.begin(), histogram.begin() + midpoint};
    vector<int> rightSide = {histogram.begin() + midpoint, histogram.end()};
    vector<int>::iterator result;
    result = max_element(leftSide.begin(), leftSide.end());
    // TODO distance is O(n) AFAIK it would be best to implement my own method
    leftMax = (int) distance(leftSide.begin(), result);
    result = max_element(rightSide.begin(), rightSide.end());
    rightMax = (int) distance(rightSide.begin(), result) + midpoint;
}

static inline void getHistogram(const Mat &src, vector<int> &hist) {
    // Compute histogram
    for (unsigned long i = 0; i < hist.size(); i++) {
        for (int j = (int)src.rows/2; j < src.rows; j++) {
            if ((int)src.at<uint8_t>(j, i) == 255)
                hist[i]++;
        }
    }
}

static inline Mat atan2Mat(const Mat &valX, const Mat &valY) {
    Mat ret(valX.rows, valX.cols, CV_64F);
    for (int i = 0; i < valX.rows; i++) {
        for (int j = 0; j < valX.cols; j++) {
            ret.at<double>(i, j) = atan2(valX.at<double>(i, j), valY.at<double>(i, j));
        }
    }
    
    return ret;
}

#endif /* detection_helper_hpp */
