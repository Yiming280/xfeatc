#pragma once

#include <opencv2/opencv.hpp>
#include <string>
#include <memory>
#include "OPTApi.h"

class OptCamera {
public:
    // Construct by device index in enumerated list (default 0)
    explicit OptCamera(unsigned index = 0);
    ~OptCamera();

    // connect to camera (create handle + open). Returns true on success
    bool connect();

    // disconnect and free resources
    void disconnect();

    // start/stop grabbing stream
    bool startGrabbing();
    bool stopGrabbing();

    // capture one image synchronously (will block up to timeout_ms)
    // returns empty Mat on failure
    cv::Mat captureImage(int timeout_ms = 500);

    // best-effort setter for exposure. Some SDKs expose control via properties;
    // this is a stub that returns true for now (user may extend using SDK property API).
    bool setExposureTime(int64_t exposure_us);

    bool isConnected() const { return handle_ != nullptr && OPT_IsOpen(handle_); }

private:
    unsigned cameraIndex_;
    OPT_HANDLE handle_;
    bool isGrabbing_;
};
