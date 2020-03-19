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
        void parseFrame(Mat &frame, bool exportFrame);
        bool init();
        float getFps();
        int getMissedFrames() const {return missedFrames;}
    private:
        // methods
        void saveFrameToFile(Mat &frame);
        vector<Vec4i> detectLines();

        // attributes
        int frameId;
        int missedFrames;
        Mat givenFrame;
        FrameConfig *cfg;
        Net net;
        vector<String> outNames;
        ObjectDetector *objectDetector;
        LineDetector *lineDetector;
        TickMeter tm;
};