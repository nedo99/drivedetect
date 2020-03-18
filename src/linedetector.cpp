#include "linedetector.h"

#include <opencv2/imgproc.hpp>


LineDetector::LineDetector(const FrameConfig &config) {
    cfg = new FrameConfig(config);
}

bool LineDetector::init() {
    return true;
}

// https://github.com/opencv/opencv/blob/fc41c18c6f27c1ae663b2b8b561235921280174c/modules/calib3d/src/chessboard.cpp
void LineDetector::polyfit(const Mat& src_x, const Mat& src_y, Mat& dst, int order)
{
    int npoints = src_x.checkVector(1);
    int nypoints = src_y.checkVector(1);
    CV_Assert(npoints == nypoints && npoints >= order+1);
    Mat_<double> srcX(src_x), srcY(src_y);
    Mat_<double> A = Mat_<double>::ones(npoints, order + 1);
    // build A matrix
    for (int x = 0; x < npoints; ++x)
    {
        for (int y = 1; y < A.cols; ++y)
            A.at<double>(x, y) = srcX.at<double>(x) * A.at<double>(x, y - 1);
    }
    static Mat w;
    solve(A, srcY, w, DECOMP_SVD);
    w.convertTo(dst, ((src_x.depth() == CV_64F || src_y.depth() == CV_64F) ? CV_64F : CV_32F));
}

Mat LineDetector::regionOfInterests(const Mat given_image, const vector<Point> pts)
{
    vector<vector<Point> > vpts;
    vpts.push_back(pts);
    Mat mask = Mat::zeros(given_image.size(), CV_8U);
    fillPoly(mask, vpts, Scalar(255, 255, 255));
    bitwise_and(given_image, mask, mask);
    return mask;
}

vector<Vec4i> LineDetector::averageSlopeIntercept(int height, vector<Vec4i> lines, double slope_intercept)
{
    vector<Vec2f> left_fit, right_fit;
    for (uint64_t i = 0; i < lines.size(); i ++) {
        vector<int> x = {lines[i][0], lines[i][2]};
        vector<int> y = {lines[i][1], lines[i][3]};
        Mat dst;
        polyfit(Mat(x), Mat(y), dst, 1);

        if (dst.at<float>(0, 1) < 0)
            left_fit.push_back(Vec2f(dst.at<float>(0, 1), dst.at<float>(0, 0)));
        else
            right_fit.push_back(Vec2f(dst.at<float>(0, 1), dst.at<float>(0, 0)));
    }
    Scalar res = mean(left_fit);
    Vec2f left_fit_average(res[0], res[1]);
    res = mean(right_fit);
    Vec2f right_fit_average(res[0], res[1]);
    Vec4i left_line = makeCoordinates(height, left_fit_average, slope_intercept);
    Vec4i right_line = makeCoordinates(height, right_fit_average, slope_intercept);

    return {left_line, right_line};
}

Vec4i LineDetector::makeCoordinates(int height, Vec2f line_parameters, double slope_intercept)
{
    int y1 = height;
    int y2 = int(y1 * slope_intercept);
    int x1 = (int)(y1 - line_parameters[1]) / line_parameters[0];
    int x2 = (int)(y2 - line_parameters[1]) / line_parameters[0];
    return Vec4i(x1, y1, x2, y2);
}

vector<Vec4i> LineDetector::detectLines(const Mat &frame) {
    try {
        static Mat tmpFrame;
        cvtColor(frame, tmpFrame, COLOR_RGB2GRAY);
        
        static Size blur_size;
        blur_size.width = cfg->getBlurWidth();
        blur_size.height = cfg->getBlurHeight();
        GaussianBlur(tmpFrame, tmpFrame, blur_size, 0);
        if (tmpFrame.empty()) {
            cout << "blurred image empty" << endl;
        }

        Canny(tmpFrame, tmpFrame, cfg->getCannyLThr(), cfg->getCannyHThr());
        if (tmpFrame.empty()) {
            cout << "Canny call returned errors!" << endl;
        }
        
        tmpFrame = regionOfInterests(tmpFrame, cfg->getMaskPoints());
        vector<Vec4i> lines;
        HoughLinesP(tmpFrame, lines, cfg->getRho(),  cfg->getTheta(), cfg->getHlpThreshold(), cfg->getMinLineLength(), cfg->getMaxLineGap());
        if (lines.size() < 1) {
            cerr << "Could not find HoughLinesP for given frame!" << endl;
            return vector<Vec4i>();
        }

        return averageSlopeIntercept(tmpFrame.size().height, lines, cfg->getSlopeIntercept());
    }
    catch(cv::Exception& e)
    {
        cerr << "Exception caught: " << e.what() << endl;
        cfg->printConfig();
    }

    return vector<Vec4i>();
}