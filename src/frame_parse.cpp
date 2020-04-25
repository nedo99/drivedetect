#include "frame_parse.h"

#include <stdio.h>
#include <unistd.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <filesystem>
#include <thread>

#include "detection_helper.hpp"

// OpenCV related includes
#include <opencv2/imgproc.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/highgui.hpp>

static mutex threadMutex;
static deque<int> deleteThread;

Mat whiteYellowMask, shapeMsk, sobelXMask, sobelYMask, sobelXYMask, combinedMask, plotyLinespace;

static void drawLines(vector<Vec4i> lines, Mat &dst_image);
void drawRectangles(const vector<int> &classIds, const vector<float> &confs, const vector<Rect> &boxes, const Mat& frame,
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

void drawRectangles(const vector<int> &classIds, const vector<float> &confs, const vector<Rect> &boxes,
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

void detect(Mat frame, FrameParse *obj, int threadIndex, int frameId) {
    Mat detectedFrame, givenFrame;
    LineDetector *lineDetector = new LineDetector(obj->getConfig(), obj->getFrameWidth(), obj->getFrameHeight());

    ObjectDetector *objDetct = new ObjectDetector(*obj->cfg, obj->net);
    vector<Rect> boxes;
    if (obj->cfg->additionalImageProcessing) {
        if (obj->cfg->advCfg.scale < 1) {
            resize(frame, givenFrame, Size(obj->newWidth, obj->newHeight));
        } else {
            if (obj->calibrateFrame) {
                static InputArrayOfArrays objPts = InputArrayOfArrays(obj->objPoints);
                static InputArrayOfArrays imgPts = InputArrayOfArrays(obj->corners);
                undistortImage(frame, givenFrame, objPts, imgPts);
            }
            else
            {
                givenFrame = frame;
            }
        }

        //boxes = objDetct->detectObjects(frame);
        if (!lineDetector->advancedLineDetection(givenFrame)) {
            std::lock_guard<std::mutex> lock(threadMutex);
            obj->saveFrameToFile(frame);
            obj->missedFrames++;
        }
        obj->lastLeftCurvature = lineDetector->getLeftCurvature();
        obj->lastRightCurvature = lineDetector->getRightCurvature();
    } else {
        givenFrame = frame;
        boxes = objDetct->detectObjects(frame);
        vector<Vec4i> lines = lineDetector->detectLines(frame);
        if (lines.empty()) {
            std::lock_guard<std::mutex> lock(threadMutex);
            obj->saveFrameToFile(frame);
            obj->missedFrames++;
        }
        drawLines(lines, givenFrame);
    }
    
    drawRectangles(objDetct->getClassIds(), objDetct->getConfidences(), boxes,
        givenFrame, obj->cfg->getClasses(), objDetct->getIndices());
    std::lock_guard<std::mutex> lock(threadMutex);
    deleteThread.push_back(frameId);
    obj->frameId++;
    obj->parsedFrames[frameId] = givenFrame;
    delete lineDetector;
    delete objDetct;
}

FrameParse::FrameParse(const string cfgPath) {
    cfg = new FrameConfig(cfgPath);
    frameId = 0;
    missedFrames = 0;
    calibrateFrame = false;
    this->returnedIndex = 0;
}

float FrameParse::getFps() {
    tm.stop();
    float fps = frameId / tm.getTimeSec();
    tm.start();
    return fps;
}

Mat FrameParse::parseFrame(const Mat &frame, int frameId, bool exportFrame=false) {
    if (exportFrame)
        saveFrameToFile(frame);
    
    if (frameId == 1) {
        tm.reset();
        tm.start();
    }

    this->threads[frameId] = thread(detect, frame, this, threads.size(), frameId);
    
    if (deleteThread.size()) {
        int threadIndex = deleteThread.front();
        this->threads[threadIndex].join();
        this->threads.erase(threadIndex);
        deleteThread.pop_front();
    }

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

bool FrameParse::init(int frameWidth, int frameHeight) {
    this->frameWidth = frameWidth;
    this->frameHeight = frameHeight;
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
    
    this->net = readNet(cfg->dnn_model, cfg->dnn_config);
    this->net.setPreferableBackend(0);
    this->net.setPreferableTarget(0);
    this->initFrame(frameWidth, frameHeight);

    if (cfg->advCfg.scale < 1) {
        this->newHeight = (int)(frameHeight * cfg->advCfg.scale);
        this->newWidth = (int)(frameWidth * cfg->advCfg.scale);
    }

    if (!this->net.getUnconnectedOutLayersNames().size())
        return false;
    
    if (!cfg->advCfg.calibrationPath.empty()) {
        findImagePoints(cfg->advCfg.calibrationPath, corners, objPoints, cfg->advCfg.chessX, cfg->advCfg.chessY);
        calibrateFrame = true;
    }
    
    return true;
}

void FrameParse::initFrame(int frameWidth, int frameHeight) {
    // Optionally frame size could be written in config file
    vector<vector<Point> > vpts;
    vpts.push_back(cfg->maskPts);
    Size frameSize = Size(frameWidth, frameHeight);
    shapeMsk = Mat::zeros(frameSize, CV_8U);
    fillPoly(shapeMsk, vpts, Scalar(255, 255, 255));
    sobelXMask = Mat::zeros(frameWidth, frameHeight, CV_8U);
    sobelYMask = Mat::zeros(frameWidth, frameHeight, CV_8U);
    sobelXYMask = Mat::zeros(frameWidth, frameHeight, CV_8U);
    combinedMask = Mat::zeros(frameWidth, frameHeight, CV_8U);
    whiteYellowMask = Mat::zeros(frameWidth, frameHeight, CV_8U);
    plotyLinespace = linespace(0, frameHeight, frameHeight);
}

void FrameParse::deinit() {
    for (auto &t: this->threads) {
        if (t.second.joinable())
            t.second.join();
    }
    this->threads.clear();
    this->parsedFrames.clear();
}

Mat FrameParse::getNextParsedFrame() {
    Mat ret;
    if (this->parsedFrames.find(this->returnedIndex) != this->parsedFrames.end()) {
        ret = this->parsedFrames[this->returnedIndex];
        this->parsedFrames.erase(this->returnedIndex);
        this->returnedIndex++;
    }
    return ret;
}
