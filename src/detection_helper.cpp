//
//  detection_helper.cpp
//  
//
//  Created by Nedim Hadzic
//

#include "detection_helper.hpp"
#include <iostream>

#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

#define MINPIX           50
#define UNUSED(x)        (void)(x)

// TODO optimize
Mat getFitX(const Mat &polyfit, const Mat &ploty) {
    Mat res(ploty.cols, 1, CV_64F);
    Mat tmp(ploty.cols, 1, CV_64F);
    Mat tmp1(ploty.cols, 1, CV_64F, Scalar(polyfit.at<float>(0,0)));
    pow(ploty, 2, res);
    res = polyfit.at<float>(2, 0) * res;
    tmp = polyfit.at<float>(1, 0) * ploty;
    res = res + tmp;
    res = res + tmp1;

    return res;
}

static void getIndices(const Mat &nonZero, const Mat &leftFit, const Mat &rightFit,
                       Mat &leftX, Mat &leftY, Mat &rightX, Mat &rightY, int margin) {
    for (int i = 0; i < nonZero.rows; i++) {
        double xsum = leftFit.at<float>(2, 0) * pow(nonZero.at<Point>(i, 0).y, 2) +
        leftFit.at<float>(1, 0) * nonZero.at<Point>(i, 0).y + leftFit.at<float>(0, 0);
        double ysum = rightFit.at<float>(2, 0) * pow(nonZero.at<Point>(i, 0).y, 2) +
        rightFit.at<float>(1, 0) * nonZero.at<Point>(i, 0).y + rightFit.at<float>(0, 0);
        double xlowSum = xsum - margin;
        double xhighSum = xsum + margin;
        double ylowSum = ysum - margin;
        double yhighSum = ysum + margin;
        if ((nonZero.at<Point>(i, 0).x > xlowSum) && (nonZero.at<Point>(i, 0).x < xhighSum)) {
            leftX.push_back(nonZero.at<Point>(i, 0).x);
            leftY.push_back(nonZero.at<Point>(i, 0).y);
        }
        if ((nonZero.at<Point>(i, 0).x > ylowSum) && (nonZero.at<Point>(i, 0).x < yhighSum)) {
            rightX.push_back(nonZero.at<Point>(i, 0).x);
            rightY.push_back(nonZero.at<Point>(i, 0).y);
        }
    }
}

// TODO do better
Mat linespace(int start, int end, int samples) {
    UNUSED(start);
    UNUSED(end);
    Mat lineSpace(samples, 1, CV_64F);
    for (int i = 0; i < samples; i++)
        lineSpace.at<double>(i, 0) = (double)i;
    return lineSpace;
}

/*
void skipSlideWindow(const Mat &nonZero,
                     Mat &leftFitX, Mat &rightFitX, int margin) {
    static Mat leftX, leftY, rightX, rightY;
    getIndices(nonZero, leftFit, rightFit, leftX, leftY, rightX, rightY, margin);
    polyfit(leftY, leftX, leftFit, 2);
    polyfit(rightY, rightX, rightFit, 2);
    Mat ploty = linespace(0, 720, 720);
    leftFitX = getFitX(leftFit, ploty);
    rightFitX = getFitX(rightFit, ploty);
}
*/
// https://github.com/opencv/opencv/blob/fc41c18c6f27c1ae663b2b8b561235921280174c/modules/calib3d/src/chessboard.cpp
void polyfit(const Mat& src_x, const Mat& src_y, Mat& dst, int order) {
    int npoints = src_x.checkVector(1);
    int nypoints = src_y.checkVector(1);
    CV_Assert(npoints == nypoints && npoints >= order + 1);
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


void measeureCurvature(const Mat &ploty, const Mat &leftFitX, const Mat &rightFitX, const double &ym_per_pix, const double &xm_per_pix,
                       double &leftCurvature, double &rightCurvature) {
    Mat leftFitCurvature, rightFitCurvature, tmp, calcTmp;
    tmp = leftFitX * xm_per_pix;
    calcTmp = ploty * ym_per_pix;
    polyfit(calcTmp, tmp, leftFitCurvature, 2);
    tmp = rightFitX * xm_per_pix;
    polyfit(calcTmp, tmp, rightFitCurvature, 2);
    double maxPloty, minPloty;
    minMaxLoc(ploty, &minPloty, &maxPloty);

    double val1 = 2.0 * leftFitCurvature.at<double>(2, 0) * maxPloty * ym_per_pix + leftFitCurvature.at<double>(1, 0);
    double val2 = 2.0 * rightFitCurvature.at<double>(2, 0) * maxPloty * ym_per_pix + rightFitCurvature.at<double>(1, 0);
    val1 = pow(val1, 2) + 1;
    val1 = pow(val1, 1.5);
    val2 = pow(val2, 2) + 1;
    val2 = pow(val2, 1.5);
    leftCurvature = val1 / abs(2 * leftFitCurvature.at<double>(2, 0));
    rightCurvature = val2 / abs(2 * rightFitCurvature.at<double>(2, 0));
}

void drawLaneLines(Mat &frame, const Mat &persImg, const Mat &persFrameInv, Mat &leftFitX, Mat &rightFitX,
                   const Mat &ploty) {
    static vector<Point> pts(2 * leftFitX.rows);
    for (int i = 0; i < leftFitX.rows; i++) {
        Point leftP((int)leftFitX.at<double>(i, 0), (int)ploty.at<double>(i, 0));
        pts[i] = leftP;
        int backIndex = leftFitX.rows - 1 - i;
        Point rightP((int)rightFitX.at<double>(backIndex, 0), (int)ploty.at<double>(backIndex, 0));
        pts[i + leftFitX.rows] = rightP;
    }

    vector<vector<Point>> vpts;
    vpts.push_back(pts);
    Mat colorWarp = Mat::zeros(persImg.rows, persImg.cols, CV_8UC3);
    fillPoly(colorWarp, vpts, Scalar(0, 255, 0));
    Mat newWrap;
    warpPerspective(colorWarp, newWrap, persFrameInv, Size(frame.cols, frame.rows));
    addWeighted(frame, 1.0, newWrap, 0.3, 0, frame);
#if defined (VALIDATE_PARSING) && (VALIDATE_PARSING==1)
    imshow("colorWarp", colorWarp);
    waitKey(0);
#endif
}

void perspectiveTransform(const Mat &frame, const vector<Point2f> &srcPts, const vector<Point2f> &dstPts,
                                 Mat &outFrame, Mat &persTransformInv) {
    outFrame = getPerspectiveTransform(srcPts, dstPts);
    persTransformInv = getPerspectiveTransform(dstPts, srcPts);
    warpPerspective(frame, outFrame, outFrame, Size(frame.cols, frame.rows), INTER_LINEAR);
}

bool computeLineLanes(const Mat &persImg, const Mat &nonZero, const vector<int> &histogram,
                        Mat &leftFitX, Mat &rightFitX, int margin, int nWindows) {
    Mat leftFit, rightFit;
    int leftHistMaxIndex, rightHistMaxIndex;
    getLeftAndRightHistMaximum(histogram, leftHistMaxIndex, rightHistMaxIndex);
#if defined (VALIDATE_PARSING) && (VALIDATE_PARSING==1)
    Mat outImg(persImg.rows, persImg.cols, CV_8UC3);
    Mat in[] = {persImg, persImg, persImg};
    int from_to[] = { 0,0, 1,1, 2,2 };
    mixChannels(in, 3, &outImg, 1, from_to, 3);
#endif
    int windowHeight = (int) persImg.rows / nWindows;
    int leftx_current = leftHistMaxIndex;
    int rightx_current = rightHistMaxIndex;
    Mat leftLineInd, rightLineInd;
    for (int i = 0; i < nWindows; i ++) {
        int win_y_low = persImg.rows - (i + 1) * windowHeight;
        int win_y_high = persImg.rows - i * windowHeight;
        int win_xleft_low = leftx_current - margin;
        int win_xleft_high = leftx_current + margin;
        int win_xright_low = rightx_current - margin;
        int win_xright_high = rightx_current + margin;
#if defined (VALIDATE_PARSING) && (VALIDATE_PARSING==1)
        rectangle(outImg, Point(win_xleft_low, win_y_low), Point(win_xleft_high, win_y_high), Scalar(0, 255, 0), 2);
        rectangle(outImg, Point(win_xright_low, win_y_low), Point(win_xright_high, win_y_high), Scalar(0, 255, 0), 2);
#endif
        Scalar lowerThreshold(win_y_low, win_y_high), upperThreshold(win_xleft_low, win_xleft_high),
        xRightThreshold(win_xright_low, win_xright_high);
        Mat goodLeftInds, goodRightInds;
        float meanLeft, meanRight;
        getGoodIndices(nonZero, goodLeftInds, meanLeft, upperThreshold, lowerThreshold);
        getGoodIndices(nonZero, goodRightInds, meanRight, xRightThreshold, lowerThreshold);
        leftLineInd.push_back(goodLeftInds);
        rightLineInd.push_back(goodRightInds);
        if (goodLeftInds.rows > MINPIX)
            leftx_current = (int) meanLeft;
        if (goodRightInds.rows > MINPIX)
            rightx_current = (int) meanRight;
        
        if (leftx_current > rightx_current) {
            cerr << "Left X current is bigger than right X current" << endl;
            return false;
        }
    }

    Mat leftX, rightX, leftY, rightY;
    getFilteredArray(nonZero, leftLineInd, leftX, leftY);
    getFilteredArray(nonZero, rightLineInd, rightX, rightY);
    if (leftX.rows == 0 && leftX.cols == 0) {
        cerr << "Could not find left X indices" << endl;
        return false;
    }
    if (rightX.rows == 0 && rightX.cols == 0) {
        cerr << "Could not find left X indices" << endl;
        return false;
    }
#if defined (VALIDATE_PARSING) && (VALIDATE_PARSING==1)
    updateImg(outImg, leftY, leftX, Vec3b(255, 0, 0));
    updateImg(outImg, rightY, rightX, Vec3b(0, 0, 255));
#endif
    polyfit(leftY, leftX, leftFit, 2);
    polyfit(rightY, rightX, rightFit, 2);
    Mat plotyTmp = linespace(0, persImg.rows, persImg.rows);
    leftFitX = getFitX(leftFit, plotyTmp);
    rightFitX = getFitX(rightFit, plotyTmp);
#if defined (VALIDATE_PARSING) && (VALIDATE_PARSING==1)
    imshow("Line Detection", outImg);
    waitKey(0);
#endif
    return true;
}

void scaledSobel(const Mat &absSobelFrame, Mat &maskFrame, const Scalar &thres) {
    Mat tmpSobel, scaled;
    scaled = 255 * absSobelFrame;
    
    double min, max;
    minMaxLoc(absSobelFrame, &min, &max);
    
    scaled = scaled / max;
    inRange(scaled, thres[0], thres[1], maskFrame);
}
