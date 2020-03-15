#include "frame_parse.h"

#include <stdio.h>
#include <unistd.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/core/persistence.hpp>

#include <opencv2/highgui.hpp>

static void polyfit(const Mat& src_x, const Mat& src_y, Mat& dst, int order);
static Mat region_of_interests(const Mat given_image, const vector<Point> pts);
static vector<Vec4i> average_slope_intecept(int height, vector<Vec4i> lines, double slope_intercept);
static Vec4i make_coordinates(int height, Vec2f line_parameters, double slope_intercept);
static void draw_lines(vector<Vec4i> lines, Mat &dst_image);

// https://github.com/opencv/opencv/blob/fc41c18c6f27c1ae663b2b8b561235921280174c/modules/calib3d/src/chessboard.cpp
static void polyfit(const Mat& src_x, const Mat& src_y, Mat& dst, int order)
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
    Mat w;
    solve(A, srcY, w, DECOMP_SVD);
    w.convertTo(dst, ((src_x.depth() == CV_64F || src_y.depth() == CV_64F) ? CV_64F : CV_32F));
}

static Mat region_of_interests(const Mat given_image, const vector<Point> pts)
{
    vector<vector<Point> > vpts;
    vpts.push_back(pts);
    Mat mask = Mat::zeros(given_image.size(), CV_8U);
    fillPoly(mask, vpts, Scalar(255, 255, 255));
    bitwise_and(given_image, mask, mask);
    return mask;
}

static vector<Vec4i> average_slope_intecept(int height, vector<Vec4i> lines, double slope_intercept)
{
    vector<Vec2f> left_fit, right_fit;
    for (int i = 0; i < lines.size(); i ++) {
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
    Vec4i left_line = make_coordinates(height, left_fit_average, slope_intercept);
    Vec4i right_line = make_coordinates(height, right_fit_average, slope_intercept);

    return {left_line, right_line};
}

static Vec4i make_coordinates(int height, Vec2f line_parameters, double slope_intercept)
{
    int y1 = height;
    int y2 = int(y1 * slope_intercept);
    int x1 = (int)(y1 - line_parameters[1]) / line_parameters[0];
    int x2 = (int)(y2 - line_parameters[1]) / line_parameters[0];
    return Vec4i(x1, y1, x2, y2);
}

static void draw_lines(vector<Vec4i> lines, Mat &dst_image)
{
    for(int i = 0; i < lines.size(); i++)
    {
        line(dst_image, Point(lines[i][0], lines[i][1]),
        Point(lines[i][2], lines[i][3]), Scalar(255, 0, 0), 5);
    }
}

FrameParse::FrameParse(const string cfgPath) {
    cfg = new FrameConfig(cfgPath);
    frameId = 0;
}

FrameParse::~FrameParse() {
    
}

void FrameParse::parseFrame(Mat &frame, bool exportFrame=false) {
    if (exportFrame)
        saveFrameToFile(givenFrame);
    givenFrame = frame;
    Mat tmpFrame;
    try {
        cvtColor(givenFrame, tmpFrame, COLOR_RGB2GRAY);
        
        Size blur_size;
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
        
        tmpFrame = region_of_interests(tmpFrame, cfg->getMaskPoints());
        imshow("aa", tmpFrame);
        vector<Vec4i> lines;
        HoughLinesP(tmpFrame, lines, cfg->getRho(),  cfg->getTheta(), cfg->getHlpThreshold(), cfg->getMinLineLength(), cfg->getMaxLineGap());
        if (lines.size() < 1) {
            cerr << "Could not find HoughLinesP for given frame!" << endl;
            saveFrameToFile(givenFrame);
            return;
        }
        vector<Vec4i> averaged_lines = average_slope_intecept(tmpFrame.size().height, lines, cfg->getSlopeIntercept());
        draw_lines(averaged_lines, givenFrame);
        frameId++;
    }
    catch(cv::Exception& e)
    {
        const char* err_msg = e.what();
        cerr << "Exception caught: " << err_msg << endl;
        cfg->printConfig();
        saveFrameToFile(givenFrame);
    }
}

void FrameParse::saveFrameToFile(Mat &frame) {
    char filename[MAX_LOG_FILENAME];
    char currentDir[MAX_LOG_FILENAME];
    getcwd(currentDir, MAX_LOG_FILENAME);
    snprintf(filename, MAX_LOG_FILENAME, "%s/%s/frame%d.jpg", currentDir, LOG_OUTPUT, frameId);
    int result;
    try
    {
        result = imwrite(string(filename), frame);
    }
    catch (const cv::Exception& ex)
    {
        cerr << "Could not save the frame!" << ex.what() << endl;
    }
    if (result)
        cout << "Frame saved at " << filename << endl;
    else
        cerr << "Could not save the frame " << filename << endl;
}

bool FrameParse::init() {
    int ret = mkdir(LOG_OUTPUT, 0777);
    if (ret == -1) {
        if (errno != EEXIST) {
            cerr << "Could not create a log directory!" << strerror(errno) << endl;
            return false;
        }
    }
    if (cfg->parseConfig())
        return true;
    
    return false;
}