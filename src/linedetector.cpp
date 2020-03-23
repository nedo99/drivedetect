#include "linedetector.h"

#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>


LineDetector::LineDetector(const FrameConfig &config) {
    cfg = new FrameConfig(config);
    frameInitialized = false;
    gammaArray = Mat(1, 256, CV_8UC1);
}

bool LineDetector::init() {
    blur_size.width = cfg->getBlurWidth();
    blur_size.height = cfg->getBlurHeight();
    whiteLowerBound = Scalar(0, 200, 0);
    whiteUpperBound = Scalar(255, 255, 255);
    yellowLowerBound = Scalar(10, 0, 100);
    yellowUpperBound = Scalar(40, 255, 255);

    uchar * ptr = gammaArray.ptr();
    for( int i = 0; i < GAMMA_LIMIT; i++ )
        ptr[i] = (int)( pow( (double) i / 255.0, cfg->gammaConf ) * 255.0 );  

    return true;
}

void LineDetector::initFrame(const Mat &frame) {
    vector<vector<Point> > vpts;
    vpts.push_back(cfg->maskPts);
    frameSize = frame.size();
    shapeMsk = Mat::zeros(frameSize, CV_8U);
    fillPoly(shapeMsk, vpts, Scalar(255, 255, 255));
    frameY1 = frameSize.height;
    frameY2 = int(frameY1 * cfg->slopeIntercept);
    frameInitialized = true;
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

vector<Vec4i> LineDetector::averageSlopeIntercept(vector<Vec4i> lines)
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
    Vec4i left_line = makeCoordinates(left_fit_average);
    Vec4i right_line = makeCoordinates(right_fit_average);

    return {left_line, right_line};
}

Vec4i LineDetector::makeCoordinates(Vec2f line_parameters)
{
    int x1 = (int)(frameY1 - line_parameters[1]) / line_parameters[0];
    int x2 = (int)(frameY2 - line_parameters[1]) / line_parameters[0];
    return Vec4i(x1, frameY1, x2, frameY2);
}

vector<Vec4i> LineDetector::detectLines(const Mat &frame) {
    if (!frameInitialized)
        initFrame(frame);
    try {
        static Mat tmpFrame;

        cvtColor(frame, tmpFrame, COLOR_RGB2GRAY);
        
        if (cfg->additionalImageProcessing) {
            LUT(tmpFrame, gammaArray, tmpFrame);
    
            inRange(frame, whiteLowerBound, whiteUpperBound, whiteMask);

            inRange(frame, yellowUpperBound, yellowUpperBound, yellowMask);

            bitwise_or(whiteMask, yellowMask, clrMask);

            bitwise_and(tmpFrame, clrMask, tmpFrame);
        }

        GaussianBlur(tmpFrame, tmpFrame, blur_size, 0);

        Canny(tmpFrame, tmpFrame, cfg->getCannyLThr(), cfg->getCannyHThr());
        //imshow("tmp", tmpFrame);
        //waitKey(0);
        bitwise_and(tmpFrame, shapeMsk, tmpFrame);
        //imshow("tmp", tmpFrame);
        //waitKey(0);

        vector<Vec4i> lines;
        HoughLinesP(tmpFrame, lines, cfg->getRho(),  cfg->getTheta(), cfg->getHlpThreshold(), cfg->getMinLineLength(), cfg->getMaxLineGap());
        if (lines.size() < 1) {
            cerr << "Could not find HoughLinesP for given frame!" << endl;
            return vector<Vec4i>();
        }

        return averageSlopeIntercept(lines);
    }
    catch(cv::Exception& e)
    {
        cerr << "Exception caught: " << e.what() << endl;
        cfg->printConfig();
    }

    return vector<Vec4i>();
}