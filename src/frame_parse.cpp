#include "frame_parse.h"

#include <stdio.h>
#include <unistd.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <filesystem>

// OpenCV related includes
#include <opencv2/imgproc.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/highgui.hpp>

static void drawLines(vector<Vec4i> lines, Mat &dst_image);
static void drawRectangles(const vector<int> &classIds, const vector<float> &confs, const vector<Rect> &boxes, const Mat& frame,
    vector<string> classes, const vector<int> &indices);
static void undistortImage(const Mat &frame, Mat &outFrame, InputArrayOfArrays objPts, InputArrayOfArrays imgPts);
static void findImagePoints(string folderPath, vector<vector<Point2f>> &corners, vector<Mat> &objPoints,
    int rows, int cols);
static Mat prepareObjPoints(int rows, int cols);
static vector<Point2f> findChessCorners(Mat &gryImage, Size patternSize);

static vector<Point2f> findChessCorners(Mat &gryImage, Size patternSize) {
    vector<Point2f> corners; //this will be filled by the detected corners

    bool patternfound = findChessboardCorners(gryImage, patternSize, corners, 0);
    if(patternfound)
        cornerSubPix(gryImage, corners, Size(11, 11), Size(-1, -1),
            TermCriteria(TermCriteria::EPS + TermCriteria::MAX_ITER, 30, 0.1));

    //drawChessboardCorners(frame, chesseSize, Mat(corners), patternfound);
    return corners;
}

static Mat prepareObjPoints(int rows, int cols) {
    Mat val = Mat::zeros((rows * cols), 3, CV_32FC1);
    for (int i = 0; i < cols; i ++) {
        for (int j = 0; j < rows; j++) {
            int r = i * rows + j;
            val.at<float>(r, 0) = j;
            val.at<float>(r, 1) = i;
        }
    }
    return val;
}

static void findImagePoints(string folderPath, vector<vector<Point2f>> &corners, vector<Mat> &objPoints,
    int rows, int cols) {
    Mat objPoint = prepareObjPoints(rows, cols);
    for(auto& p: filesystem::directory_iterator(folderPath)) {
        Mat image = imread(p.path());
        cvtColor(image, image, COLOR_RGB2GRAY);
        vector<Point2f> c = findChessCorners(image, Size(rows, cols));
        if (c.size() != (uint32_t)objPoint.rows)
            cerr << "Object points (" << objPoint.size() << ") do not match to image points (" << c.size() << ")!" << endl;
        else {
            corners.push_back(findChessCorners(image, Size(9,6)));
            objPoints.push_back(objPoint);
        }
    }
}

static void undistortImage(const Mat &frame, Mat &outFrame, InputArrayOfArrays objPts, InputArrayOfArrays imgPts) {
    Mat cameraMatrix, distCoeffs, rvecs, tvecs;
    calibrateCamera(objPts, imgPts, frame.size(), cameraMatrix, distCoeffs, rvecs, tvecs);
    undistort(frame, outFrame,  cameraMatrix, distCoeffs);
}

static void drawRectangles(const vector<int> &classIds, const vector<float> &confs, const vector<Rect> &boxes,
    const Mat& frame, vector<string> classes, const vector<int> &indices)
{
    for (size_t i = 0; i < indices.size(); ++i)
    {
        int idx = indices[i];
        Rect box = boxes[idx];
        float conf = confs[idx];
        int classId = classIds[idx];
        rectangle(frame, Point(box.x, box.y), Point(box.x + box.width, box.y + box.height), Scalar(0, 255, 0));

        std::string label = format("%.2f", conf);
        if (!classes.empty())
        {
            CV_Assert(classId < (int)classes.size());
            label = classes[classId] + ": " + label;
        }

        int baseLine;
        Size labelSize = getTextSize(label, FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);

        int top = max(box.y, labelSize.height);
        rectangle(frame, Point(box.x, top - labelSize.height),
                Point(box.x + labelSize.width, top + baseLine), Scalar::all(255), FILLED);
        putText(frame, label, Point(box.x, top), FONT_HERSHEY_SIMPLEX, 0.5, Scalar());
    }
}

static void drawLines(vector<Vec4i> lines, Mat &dst_image)
{
    for(uint64_t i = 0; i < lines.size(); i++)
    {
        line(dst_image, Point(lines[i][0], lines[i][1]),
        Point(lines[i][2], lines[i][3]), Scalar(0, 255, 0), 4);
    }
}

FrameParse::FrameParse(const string cfgPath) {
    cfg = new FrameConfig(cfgPath);
    frameId = 0;
    missedFrames = 0;
    calibrateFrame = false;
}

float FrameParse::getFps() {
    tm.stop();
    float fps = frameId / tm.getTimeSec();
    tm.start();
    return fps;
}

Mat FrameParse::parseFrame(const Mat &frame, bool exportFrame=false) {
    if (exportFrame)
        saveFrameToFile(frame);
    
    if (frameId == 1) {
        tm.reset();
        tm.start();
    }
    vector<Rect> boxes;
    if (cfg->additionalImageProcessing) {
        if (cfg->advCfg.scale < 1) {
            int newHeight = (int)(frame.rows * cfg->advCfg.scale);
            int newWidth = (int)(frame.cols * cfg->advCfg.scale);
            resize(frame, givenFrame, Size(newWidth, newHeight));
        } else {
            if (this->calibrateFrame) {
                static InputArrayOfArrays objPts = InputArrayOfArrays(objPoints);
                static InputArrayOfArrays imgPts = InputArrayOfArrays(corners);
                undistortImage(frame, givenFrame, objPts, imgPts);
            }
            else
            {
                givenFrame = frame;
            }
        }
        boxes = objectDetector->detectObjects(givenFrame);
        
        if (!lineDetector->advancedLineDetection(givenFrame)) {
            saveFrameToFile(frame);
            missedFrames++;
        }
    } else {
        givenFrame = frame;
        boxes = objectDetector->detectObjects(frame);
        vector<Vec4i> lines = lineDetector->detectLines(frame);
        if (lines.empty()) {
            saveFrameToFile(frame);
            missedFrames++;
        }
        drawLines(lines, givenFrame);
    }
    
    drawRectangles(objectDetector->getClassIds(), objectDetector->getConfidences(), boxes,
        givenFrame, cfg->getClasses(), objectDetector->getIndices());
    frameId++;
    return givenFrame;
}

void FrameParse::saveFrameToFile(const Mat &frame) {
    string filename = absLogPath + "/frame_" + to_string(frameId) + ".jpg";
    int result;
    try
    {
        result = imwrite(filename, frame);
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
    absLogPath = filesystem::current_path().string() + '/' + LOG_OUTPUT;
    int ret = mkdir(absLogPath.c_str(), 0777);
    if (ret == -1) {
        if (errno != EEXIST) {
            cerr << "Could not create a log directory!" << strerror(errno) << endl;
            return false;
        }
    }
    if (!cfg->parseConfig())
        return false;
    
    objectDetector = new ObjectDetector(*cfg);
    if (!objectDetector->init())
        return false;
    
    lineDetector = new LineDetector(*cfg);
    if (!lineDetector->init())
        return false;
    
    if (!cfg->advCfg.calibrationPath.empty()) {
        findImagePoints(cfg->advCfg.calibrationPath, corners, objPoints, cfg->advCfg.chessX, cfg->advCfg.chessY);
        calibrateFrame = true;
    }
    
    return true;
}

double FrameParse::getLastLeftCurvature() const {
    if (lineDetector)
        return lineDetector->getLeftCurvature();
    return 0;
}

double FrameParse::getLastRightCurvature() const {
    if (lineDetector)
        return lineDetector->getRightCurvature();
    return 0;
}
