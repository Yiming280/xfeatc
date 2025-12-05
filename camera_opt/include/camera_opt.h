
#pragma once
#include <opencv2/opencv.hpp>

#ifdef CAMERAOPT_EXPORTS
#define CAMERAOPT_API __declspec(dllexport)
#else
#define CAMERAOPT_API __declspec(dllimport)
#endif

class CAMERAOPT_API CameraOpt{
public:
    CameraOpt();
    ~CameraOpt();

    // Initialize camera (returns true if successful)
    bool initialize(int deviceID = 0);

    // Grab a frame (grayscale)
    bool grabFrame(cv::Mat& frame);

    // Release camera
    void release();

private:
    cv::VideoCapture cap_;
};
