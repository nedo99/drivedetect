#include "frame_parse.h"

#include <stdio.h>
#include <unistd.h>
#include <sys/stat.h>
#include <sys/types.h>

// OpenCV related includes
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

static void draw_lines(vector<Vec4i> lines, Mat &dst_image);
static void drawRectangles(const vector<int> &classIds, const vector<float> &confs, const vector<Rect> &boxes, const Mat& frame,
    vector<string> classes, const vector<int> &indices);

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

static void draw_lines(vector<Vec4i> lines, Mat &dst_image)
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
}

FrameParse::~FrameParse() {
    
}

void FrameParse::parseFrame(Mat &frame, bool exportFrame=false) {
    givenFrame = frame;
    if (exportFrame)
        saveFrameToFile(givenFrame);

    vector<Rect> boxes = objectDetector->detectObjects(givenFrame);
    vector<Vec4i> lines = lineDetector->detectLines(givenFrame);
    if (lines.empty())
        saveFrameToFile(givenFrame);
    drawRectangles(objectDetector->getClassIds(), objectDetector->getConfidences(), boxes,
        givenFrame, cfg->getClasses(), objectDetector->getIndices());
    draw_lines(lines, givenFrame);
    frameId++;
}

void FrameParse::saveFrameToFile(Mat &frame) {
    char filename[MAX_LOG_FILENAME];
    char currentDir[LOG_FILENAME];
    getcwd(currentDir, MAX_LOG_FILENAME);
    snprintf(filename, MAX_LOG_FILENAME, "%s/%s/frame%d.jpg", currentDir, LOG_OUTPUT, frameId);
    int result;
    try
    {
        result = imwrite(string(filename), frame);
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
    int ret = mkdir(LOG_OUTPUT, 0777);
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
    
    return true;
}


    /*
    string label = format("Camera: %.2f FPS", 15);
    putText(frame, label, Point(0, 15), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 255, 0));

    label = format("Network: %.2f FPS", 15);
    putText(frame, label, Point(0, 30), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 255, 0));

    label = format("Skipped frames: %d", 6);
    putText(frame, label, Point(0, 45), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 255, 0));
    imshow(kWinName, frame);
    // Press  ESC on keyboard to exit
    char c=(char)waitKey(25);
    if(c==27)
        break;
    */
