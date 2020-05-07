#include <string>
#include <iostream>
#include <chrono>
#include <map>
#include <cstdlib>
#include <filesystem>

#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include "frame_parse.h"
#include "frame_config.h"

#define CONF_KEY  "config"
#define INPUT_KEY "input"

using namespace std;

static map<string, string> cmd_parameters = {
    { CONF_KEY, "" },
    { INPUT_KEY, "" }
};

static bool export_frames = false;

static void show_usage(std::string name)
{
    cerr << "Usage: " << name << " --config <path> --input <path>"
              << endl << "Options:\n"
              << "\t-h,--help\t\tShow this help message" << endl
              << "\t-c,--config \tSpecify the configuration file path" << endl
              << "\t-i,--input \tSpecify the path to image or video file" << endl
              << "\t-e, export \tWill export every parsed frame"
              << endl;
}

static bool file_exists(const string file_path) {
    if (!std::filesystem::exists(file_path)) {
        cerr << "File " << file_path << " does not exists!" << endl;
        return false;
    }

    return true;
}

static bool cmd_args_parser(int argc, char* argv[]) {
    if (argc < 3) {
        show_usage(argv[0]);
        return false;
    }

    for (int i = 1; i < argc; ++i) {
        string arg = argv[i];
        if ((arg == "-h") || (arg == "--help")) {
            show_usage(argv[0]);
            return 0;
        } else if ((arg == "-c") || (arg == "--config")) {
            if (i + 1 < argc) {
                cmd_parameters[CONF_KEY] = argv[++i];
                if (!file_exists(cmd_parameters[CONF_KEY]))
                    return false;
            } else {
                cerr << "--config option requires one argument." << endl;
                return false;
            }  
        } else if ((arg == "-i") || (arg == "--input")) {
            if (i + 1 < argc) {
                cmd_parameters[INPUT_KEY] = argv[++i];
                if (!file_exists(cmd_parameters[INPUT_KEY]))
                    return false;
            } else { // Uh-oh, there was no argument to the destination option.
                cerr << "--input option requires one argument." << endl;
                return false;
            }  
        } else if ((arg == "-e") || (arg == "--export")) {
            export_frames = true; 
        } else {
            show_usage(argv[0]);
            return false;
        }
    }

    return true;
}

static void readVideo(VideoCapture &cap) {
    FrameParse m(cmd_parameters[CONF_KEY]);
    Mat frame, detectedFrame;
    auto start = chrono::steady_clock::now();
    auto framesCount = cap.get(CAP_PROP_FRAME_COUNT);
    auto fps = cap.get(CAP_PROP_FPS);
    auto duration = framesCount / fps;
    auto frameWidth = cap.get(CAP_PROP_FRAME_WIDTH);
    auto frameHeight = cap.get(CAP_PROP_FRAME_HEIGHT);
    cout << "------- Video information ---" << endl
    << "Total frames: " << framesCount << endl
    << "Video FPS: " << fps << endl
    << "Frame size: " << frameHeight << "x" << frameWidth << endl
    << "Duration: " << duration << " secs" << endl;
    m.parseVideo(cap, export_frames);

    auto end = chrono::steady_clock::now();
    cout << "-----------------------" << endl;
    cout << "Execution time: " << chrono::duration_cast<chrono::seconds>(end - start).count() << " secs" << endl;
    cout << "Total frames: " << m.getFramesCount() << endl;
    cout << "Missed frames: " << m.getMissedFrames() << endl;
    if (m.getMissedFrames())
        cout << "Log path with missed frames: " << m.getLogPath() << endl;
    cout << "Average FPS: " << m.getFps() << endl;
    cout << "-----------------------" << endl;
}

static void readImage(Mat frame) {
    FrameParse m(cmd_parameters[CONF_KEY]);
    if (!m.init(frame.cols, frame.rows))
        return;
    imshow("win", m.parseFrame(frame, 1, export_frames));
    waitKey(0);
}

int main(int argc, char* argv[])
{
    if (!cmd_args_parser(argc, argv))
        return 1;
    Mat givenImage;
    givenImage = imread(cmd_parameters[INPUT_KEY]);

    if (givenImage.empty())
    {
        VideoCapture videoCap;
        if (videoCap.open(cmd_parameters[INPUT_KEY])) {
            readVideo(videoCap);
        } else {
            cerr << "Provided input file is not an image nor a video!" << endl;
            return 1;
        }
        
    } else {
        readImage(givenImage);
    }
    destroyAllWindows();
    return 0;
}

