#include "OptCamera.h"
#include <iostream>
#include <cstring>
#include <thread>
#include <fstream>
#include <sstream>
#include <map>
#include <algorithm>

OptCamera::OptCamera(unsigned index)
    : cameraIndex_(index), handle_(nullptr), isGrabbing_(false) {}

OptCamera::~OptCamera() {
    disconnect();
}

bool OptCamera::connect() {
    if (handle_ != nullptr) return true;

    // Enumerate devices first (the SDK examples call OPT_EnumDevices before creating a handle)
    OPT_DeviceList devList{};
    int ret = OPT_EnumDevices(&devList, interfaceTypeAll);
    if (ret != OPT_OK) {
        std::cerr << "OPT_EnumDevices failed: " << ret << std::endl;
        return false;
    }

    if (devList.nDevNum == 0) {
        std::cerr << "No devices found by OPT_EnumDevices" << std::endl;
        return false;
    }

    if (cameraIndex_ >= devList.nDevNum) {
        std::cerr << "Requested camera index " << cameraIndex_ << " out of range (found " << devList.nDevNum << ")" << std::endl;
        return false;
    }

    // Create handle using an unsigned int index variable (SDK expects pointer to unsigned int)
    unsigned int idx = cameraIndex_;
    ret = OPT_CreateHandle(&handle_, modeByIndex, (void*)&idx);
    if (ret != OPT_OK) {
        std::cerr << "OPT_CreateHandle failed: " << ret << std::endl;
        handle_ = nullptr;
        return false;
    }

    ret = OPT_Open(handle_);
    if (ret != OPT_OK) {
        std::cerr << "OPT_Open failed: " << ret << std::endl;
        OPT_DestroyHandle(handle_);
        handle_ = nullptr;
        return false;
    }

    return true;
}

void OptCamera::disconnect() {
    if (handle_ == nullptr) return;

    if (isGrabbing_) {
        OPT_StopGrabbing(handle_);
        isGrabbing_ = false;
    }

    OPT_Close(handle_);
    OPT_DestroyHandle(handle_);
    handle_ = nullptr;
}

bool OptCamera::startGrabbing() {
    if (!isConnected()) return false;
    int ret = OPT_StartGrabbing(handle_);
    if (ret != OPT_OK) {
        std::cerr << "OPT_StartGrabbing failed: " << ret << std::endl;
        return false;
    }
    isGrabbing_ = true;
    return true;
}

bool OptCamera::stopGrabbing() {
    if (!isConnected() || !isGrabbing_) return false;
    int ret = OPT_StopGrabbing(handle_);
    if (ret != OPT_OK) {
        std::cerr << "OPT_StopGrabbing failed: " << ret << std::endl;
        return false;
    }
    isGrabbing_ = false;
    return true;
}

cv::Mat OptCamera::captureImage(int timeout_ms) {
    cv::Mat out;
    if (!isConnected()) return out;

    OPT_Frame frame;
    int ret = OPT_GetFrame(handle_, &frame, timeout_ms);
    if (ret != OPT_OK) {
        // timeout or error
        return out;
    }

    // The SDK frame structure is expected to provide image buffer pointer and frame info
    // Try to read commonly available fields; this is a minimal implementation that
    // handles common pixel types (Mono8, BGR8, RGB8). The frame buffer is copied
    // into an OpenCV Mat and then SDK frame is released.

    unsigned int width = 0, height = 0;
    int pixelType = 0;
    unsigned char* buf = nullptr;

    // Defensive extraction: many SDKs expose frame.frameInfo.width/height/pixelType and pImgBuf
    // We'll attempt access and fallback conservatively if unavailable at compile-time.
    // Note: headers in camera_opt/include provide the real definitions; here we assume names.
    width = (unsigned int)frame.frameInfo.width;    // 图像宽度	 
    height = (unsigned int)frame.frameInfo.height;  // 图像高度	
    pixelType = (int)frame.frameInfo.pixelFormat;   // 图像像素格式		
    buf = (unsigned char*)frame.pData;              // 帧图像数据的内存首地址	

    if (buf == nullptr || width == 0 || height == 0) {
        OPT_ReleaseFrame(handle_, &frame);
        return out;
    }

    if (pixelType == gvspPixelMono8) {
        // single channel
        cv::Mat tmp((int)height, (int)width, CV_8UC1, buf);
        out = tmp.clone();
    }
    else if (pixelType == gvspPixelBGR8) {
        cv::Mat tmp((int)height, (int)width, CV_8UC3, buf);
        out = tmp.clone();
    }
    else if (pixelType == gvspPixelRGB8) {
        cv::Mat tmp((int)height, (int)width, CV_8UC3, buf);
        cv::cvtColor(tmp, out, cv::COLOR_RGB2BGR);
    }
    else {
        // Unsupported pixel format: try to treat as mono8
        cv::Mat tmp((int)height, (int)width, CV_8UC1, buf);
        out = tmp.clone();
    }

    OPT_ReleaseFrame(handle_, &frame);
    return out;
}

bool OptCamera::setExposureTime(int64_t exposure_us) {
    // Many GenICam-style SDKs expose property set APIs (e.g., OPT_SetInt/OPT_SetDouble).
    // The open header `OPTApi.h` in this workspace may contain such functions, but they
    // are omitted from supplied attachments. For now this function is a stub so the
    // demo compiles; extend it to call the SDK property-set API if available.
    (void)exposure_us;
    return true;
}
