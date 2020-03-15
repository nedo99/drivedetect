#include <iostream>
#include <opencv2/core.hpp>

#include "frame_config.h"

#define LOG_OUTPUT       "log"
#define MAX_LOG_FILENAME 64

using namespace std;
using namespace cv;

class FrameParse {
    public:
        FrameParse(const string cfgPath);
        ~FrameParse();
        void parseFrame(Mat &frame, bool exportFrame);
        bool init();
    private:
        void saveFrameToFile(Mat &frame);
        int frameId;
        Mat givenFrame;
        FrameConfig *cfg;
};