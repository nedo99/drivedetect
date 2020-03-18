#ifndef OBJECTDETECTOR_H
#define OBJECTDETECTOR_H
#include <iostream>

#include <opencv2/core.hpp>
#include <opencv2/dnn.hpp>

#include "frame_config.h"

using namespace std;
using namespace cv;
using namespace dnn;

class ObjectDetector {
    public:
        ObjectDetector(const FrameConfig &config);

        vector<Rect> detectObjects(const Mat &frame);
        vector<int> getClassIds() const {return classIds;}
        vector<float> getConfidences() const {return confidences;}
        vector<int> getIndices() const {return indices;}
        bool init();
    private:
        // Methods
        void preprocess(const Mat& frame);
        vector<Rect> postprocess(const Mat& frame, const vector<Mat>& outs);
        
        // attributes
        FrameConfig *cfg;
        Net net;
        vector<int> classIds;
        vector<float> confidences;
        vector<int> indices;
        vector<String> outNames;
};
#endif