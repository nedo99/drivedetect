#include <iostream>
#include <unordered_map>
#include <thread>

#include <opencv2/core.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/highgui.hpp>

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
    Mat parseFrame(const Mat &frame, int frameId, bool exportFrame);
    void parseVideo(VideoCapture &cap, bool exportFrame);
    Mat getNextParsedFrame();
    bool init(int frameWidth, int frameHeight);
    void deinit();
    float getFps();
    uint64_t getMissedFrames() const {return missedFrames;}
    uint64_t getFramesCount() const {return frameId;}
    string getLogPath() const {return absLogPath;}
    double getLastLeftCurvature() const {return lastLeftCurvature;}
    double getLastRightCurvature() const {return lastRightCurvature;}
    FrameConfig getConfig() const {return *this->cfg;}
    int getFrameWidth() const {return this->frameWidth;}
    int getFrameHeight() const {return this->frameHeight;}

    // methods
    void saveFrameToFile(const Mat &frame);
    vector<Vec4i> detectLines();
    void initFrame(int frameWidth, int frameHeight);
    
    // attributes
    uint64_t frameId;
    uint64_t missedFrames;
    Mat givenFrame, scaledFrame;
    FrameConfig *cfg;
    Net net;
    TickMeter tm;
    vector<vector<Point2f>> corners;
    vector<Mat> objPoints;
    bool calibrateFrame;
    string absLogPath;
    int frameWidth, frameHeight, newWidth, newHeight, returnedIndex;
    double lastLeftCurvature, lastRightCurvature;
    unordered_map<int, Mat> parsedFrames;
    unordered_map<int, thread> threads;
    uint32_t threadCount;
    private:
};
