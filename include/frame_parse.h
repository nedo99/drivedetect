#include <iostream>

#include <opencv2/core.hpp>
#include <opencv2/dnn.hpp>

#include "frame_config.h"
#include "objectdetector.h"
#include "linedetector.h"

#define LOG_OUTPUT       "log"
#define LOG_FILENAME     32
#define MAX_LOG_FILENAME 64

using namespace std;
using namespace cv;
using namespace dnn;

class FrameParse {
public:
    FrameParse(const string cfgPath);
    Mat parseFrame(const Mat &frame, bool exportFrame);
    void calibrateCamMatrix(const Size &frameSize);
    bool init();
    float getFps();
    uint64_t getMissedFrames() const {return missedFrames;}
    uint64_t getFramesCount() const {return frameId;}
    string getLogPath() const {return absLogPath;}
    double getLastLeftCurvature() const;
    double getLastRightCurvature() const;
private:
    // methods
    void saveFrameToFile(const Mat &frame);
    vector<Vec4i> detectLines();
    
    // attributes
    uint64_t frameId;
    uint64_t missedFrames;
    Mat givenFrame, scaledFrame, cameraMatrix, distCoeffs;
    FrameConfig *cfg;
    Net net;
    vector<String> outNames;
    ObjectDetector *objectDetector;
    LineDetector *lineDetector;
    TickMeter tm;
    vector<vector<Point2f>> corners;
    vector<Mat> objPoints;
    bool calibrateFrame;
    string absLogPath;
};
