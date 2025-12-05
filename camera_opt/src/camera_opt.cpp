#define CAMERAOPT_EXPORTS
#include "camera_opt.h"
#include <iostream>


CameraOpt::CameraOpt() {}

CameraOpt::~CameraOpt() {
    release();
}

bool CameraOpt::initialize(int deviceID) {
    cap_.open(deviceID);
    if (!cap_.isOpened()) {
        std::cerr << "Failed to open camera device " << deviceID << std::endl;
        return false;
    }
    // Optional: set resolution
    cap_.set(cv::CAP_PROP_FRAME_WIDTH, 640);
    cap_.set(cv::CAP_PROP_FRAME_HEIGHT, 480);
    return true;
}

bool CameraOpt::grabFrame(cv::Mat& frame) {
    if (!cap_.isOpened()) return false;

    cv::Mat raw;
    if (!cap_.read(raw)) return false;

    // convert to grayscale
    if (raw.channels() == 3)
        cv::cvtColor(raw, frame, cv::COLOR_BGR2GRAY);
    else
        frame = raw;

    return true;
}

void CameraOpt::release() {
    if (cap_.isOpened())
        cap_.release();
}
