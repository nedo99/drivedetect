#include "linedetector.h"
#include "detection_helper.hpp"
#include <math.h> /* atan2 */
#include <vector>
#include <cmath>

#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

#define SLOPE_THRESHOLD  200

static vector<Mat> labChannels;
static Mat persSrcMask, persDstMask, ploty;

static void directionSobel(const Mat &absSobelX, const Mat &absSobelY, Mat &maskFrame, const Scalar &thres) {
    Mat dirXY = atan2Mat(absSobelX, absSobelY);
    inRange(dirXY, thres[0], thres[1], maskFrame);
}

LineDetector::LineDetector(const FrameConfig &config) {
    cfg = new FrameConfig(config);
    frameInitialized = false;
    gammaArray = Mat(1, 256, CV_8UC1);
}

bool LineDetector::init() {
    blur_size.width = cfg->getBlurWidth();
    blur_size.height = cfg->getBlurHeight();
    whiteLowerBound = Scalar(210, 210, 210);
    whiteUpperBound = Scalar(255, 255, 255);
    yellowLowerBound = Scalar(190, 190, 0);
    yellowUpperBound = Scalar(255, 255, 255);
    
    uchar * ptr = gammaArray.ptr();
    for( int i = 0; i < GAMMA_LIMIT; i++ )
        ptr[i] = (int)( pow( (double) i / 255.0, cfg->gammaConf ) * 255.0 );
    
    return true;
}

void LineDetector::initFrame(const Mat &frame) {
    // Optionally frame size could be written in config file
    vector<vector<Point> > vpts;
    vpts.push_back(cfg->maskPts);
    frameSize = frame.size();
    shapeMsk = Mat::zeros(frameSize, CV_8U);
    fillPoly(shapeMsk, vpts, Scalar(255, 255, 255));
    frameY1 = frameSize.height;
    frameY2 = int(frameY1 * cfg->slopeIntercept);
    sobelXMask = Mat::zeros(frame.rows, frame.cols, CV_8U);
    sobelYMask = Mat::zeros(frame.rows, frame.cols, CV_8U);
    sobelXYMask = Mat::zeros(frame.rows, frame.cols, CV_8U);
    combinedMask = Mat::zeros(frame.rows, frame.cols, CV_8U);
    whiteYellowMask = Mat::zeros(frame.rows, frame.cols, CV_8U);
    ploty = linespace(0, frame.rows, frame.rows);
    frameInitialized = true;
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
        
        GaussianBlur(tmpFrame, tmpFrame, blur_size, 0);
        
        Canny(tmpFrame, tmpFrame, cfg->getCannyLThr(), cfg->getCannyHThr());

        bitwise_and(tmpFrame, shapeMsk, tmpFrame);
        
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

void LineDetector::combBinaryThresh(const Mat &frame, const Mat &gryFrame) {
    static Mat zerosMask = Mat::zeros(frame.rows, frame.cols, CV_8U);
    static Mat tmpFrame, l_image;
    //LUT(tmpFrame, gammaArray, tmpFrame);
    
    computeWhiteYellowBinary(frame, gryFrame);
    
    cvtColor(frame, tmpFrame, COLOR_RGB2Lab);
    
    split(tmpFrame, labChannels);
    l_image = labChannels[0];
    absXYSobel(l_image);
    combineSobels();
    Mat colorBinary(combinedMask.rows, combinedMask.cols, CV_8UC3);
    Mat in[] = {zerosMask, combinedMask, whiteYellowMask};
    int from_to[] = { 0,0, 1,1, 2,2 };
    mixChannels(in, 3, &colorBinary, 1, from_to, 3);
    bitwise_or(combinedMask, whiteYellowMask, combinedBinary);
}

bool LineDetector::advancedLineDetection(Mat &frame) {
    if (!frameInitialized)
        initFrame(frame);
    static Mat imgPers, leftFitX, rightFitX, nonZero, persimg, persImgInv, tmpFrame;
    static vector<int> histogram(frameSize.width);
    cvtColor(frame, tmpFrame, COLOR_RGB2GRAY);
    
    combBinaryThresh(frame, tmpFrame);
#if defined (VALIDATE_PARSING) && (VALIDATE_PARSING==1)
    imshow("Comb Threshold", combinedBinary);
    waitKey(0);
#endif
    perspectiveTransform(combinedBinary, cfg->advCfg.srcPts, cfg->advCfg.dstPts, imgPers);
#if defined (VALIDATE_PARSING) && (VALIDATE_PARSING==1)
    imshow("Color Threshold", imgPers);
    waitKey(0);
#endif
    getHistogram(imgPers, histogram);
    findNonZero(imgPers, nonZero);
    if (!computeLineLanes(imgPers, nonZero, histogram, leftFitX, rightFitX, cfg->advCfg.margin,
                     cfg->advCfg.nSegments))
        return false;
    //skipSlideWindow(nonZero, leftFitX, rightFitX, 100);
    measeureCurvature(ploty, leftFitX, rightFitX, cfg->advCfg.ymPerPix, cfg->advCfg.xmPerPix, leftCurvature, rightCurvature);
    double slopeLeft = leftFitX.at<double>(0, 0) - leftFitX.at<double>((leftFitX.rows - 1), 0);
    double slopeRight = rightFitX.at<double>(0, 0) - rightFitX.at<double>((rightFitX.rows - 1), 0);
    double slopeDiff = abs(slopeLeft - slopeRight);
    if (slopeDiff > SLOPE_THRESHOLD) {
        cerr << "Lines not parallel..." << endl
        << "Slope diff is " << slopeDiff << " comapring to threshold " << SLOPE_THRESHOLD << endl;
        return false;
    }
    computePerspectiveTransformMatrices(cfg->advCfg.srcPts, cfg->advCfg.dstPts, persimg, persImgInv);
    drawLaneLines(frame, imgPers, persImgInv, leftFitX, rightFitX, ploty);
    
    return true;
}

void LineDetector::computeWhiteYellowBinary(const Mat &frame, const Mat &gryFrame) {
    static Mat whiteMask, yellowMask;
    inRange(frame, whiteLowerBound, whiteUpperBound, whiteMask);
    inRange(frame, yellowUpperBound, yellowUpperBound, yellowMask);
    bitwise_or(whiteMask, yellowMask, clrMask);
    bitwise_and(gryFrame, clrMask, whiteYellowMask);
}

void LineDetector::absXYSobel(const Mat &gryFrame) {
    static Mat tmpSobelFrameXY, tmpSobelX, tmpSobelY;
    Sobel(gryFrame, this->sobelX, CV_64F, 1, 0, cfg->advCfg.combKSize);
    Sobel(gryFrame, this->sobelY, CV_64F, 0, 1, cfg->advCfg.combKSize);
    this->sobelX = abs(this->sobelX);
    this->sobelY = abs(this->sobelY);
    // First get masks for X and Y
    scaledSobel(this->sobelX, this->sobelXMask, cfg->advCfg.xThreshold);
    scaledSobel(this->sobelY, this->sobelYMask, cfg->advCfg.yThreshold);

    pow(this->sobelX, 2, tmpSobelX);
    pow(this->sobelY, 2, tmpSobelY);
    sqrt((tmpSobelX + tmpSobelY), tmpSobelFrameXY);
    scaledSobel(tmpSobelFrameXY, sobelXYMask, cfg->advCfg.xyThreshold);
}

void LineDetector::combineSobels() {
    static Mat dirMask, bitMask;
    directionSobel(this->sobelX, this->sobelY, dirMask, cfg->advCfg.angleThreshold);
    // Sobel X returned the best output so we keep all of its results. We perform a binary and on all the other sobels
    bitwise_and(this->sobelXYMask, this->sobelYMask, bitMask);
    bitwise_and(bitMask, dirMask, bitMask);
    bitwise_or(bitMask, this->sobelXMask, this->combinedMask);
}
