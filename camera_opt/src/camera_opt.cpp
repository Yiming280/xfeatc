#include <camera_opt/camera_opt.h>

namespace camera_opt {

OptCamera::OptCamera(const std::string_view camera_key)
    : is_connected_(false)
    , is_grabbing_(false)
    , return_code_(OPT_OK)
    , camera_key_(camera_key.data()) {
    SPDLOG_INFO("Enumerating devices.");
    this->return_code_ = OPT_EnumDevices(&this->device_list_, interfaceTypeAll);
    if (this->return_code_ != OPT_OK) {
        SPDLOG_ERROR("Failed to enumerate devices. Error code: {}", this->return_code_);
    }
    // print the device list
    this->enumerateDevices();
}

void OptCamera::enumerateDevices() {
    SPDLOG_INFO("Found {} devices.", this->device_list_.nDevNum);
    OPT_DeviceInfo* device_info = nullptr;
    for (int i = 0; i < this->device_list_.nDevNum; i++) {
        device_info = &this->device_list_.pDevInfo[i];
        SPDLOG_INFO("Device number = {} ", i);
        SPDLOG_INFO("Vendor name = {}", device_info->vendorName);
        SPDLOG_INFO("Model name = {}", device_info->modelName);
        SPDLOG_INFO("Serial number = {}", device_info->serialNumber);
        SPDLOG_INFO("=============================================");
    }
}

bool OptCamera::connect() {
    if (this->device_list_.nDevNum == 0) {
        SPDLOG_ERROR("No devices found.");
    }

    this->return_code_ = OPT_CreateHandle(&this->camera_handle_, modeByCameraKey, (void*)this->camera_key_.data());
    if (this->return_code_ != OPT_OK) {
        SPDLOG_ERROR("Failed to create handle. Error code: {}", this->return_code_);
        return false;
    }

    this->return_code_ = OPT_Open(this->camera_handle_);
    if (this->return_code_ != OPT_OK) {
        SPDLOG_ERROR("Failed to open camera. Error code: {}", this->return_code_);
        return false;
    }
    this->is_connected_ = true;
    this->startGrabbing();
    return true;
}

bool OptCamera::startGrabbing() {
    if (!this->is_connected_) {
        SPDLOG_ERROR("Camera is not connected.");
        return false;
    }
    this->return_code_ = OPT_StartGrabbing(this->camera_handle_);
    if (this->return_code_ != OPT_OK) {
        SPDLOG_ERROR("Failed to start grabbing. Error code: {}", this->return_code_);
        return false;
    }

    this->is_grabbing_ = true;

    this->return_code_ = OPT_SetEnumFeatureSymbol(this->camera_handle_, "TriggerMode", "Off");
    if (this->return_code_ != OPT_OK) {
        SPDLOG_ERROR("Failed to set trigger mode.");
        return false;
    }
    return true;
}

cv::Mat OptCamera::captureImage() {
    if (!this->is_connected_ && !this->is_grabbing_) {
        SPDLOG_ERROR("Camera is not connected. Cannot capture image.");
        return cv::Mat();
    }

    this->return_code_ = OPT_GetFrame(this->camera_handle_, &this->frame_, 5000);
    if (this->return_code_ != OPT_OK) {
        SPDLOG_WARN("Failed to get frame. Error code: {}", this->return_code_);
        return cv::Mat();
    }

    cv::Mat image =
        cv::Mat(this->frame_.frameInfo.height, this->frame_.frameInfo.width, CV_8UC1, this->frame_.pData).clone();

    this->return_code_ = OPT_ReleaseFrame(this->camera_handle_, &this->frame_);
    if (this->return_code_ != OPT_OK) {
        SPDLOG_WARN("Failed to release frame. Error code: {}", this->return_code_);
        return cv::Mat();
    }
    return image;
}

bool OptCamera::setExposureTime(const double exposure_time) {
    if (!this->is_connected_) {
        SPDLOG_ERROR("Camera is not connected.");
        return false;
    }

    this->return_code_ = OPT_SetDoubleFeatureValue(this->camera_handle_, "ExposureTime", exposure_time);
    if (this->return_code_ != OPT_OK) {
        SPDLOG_ERROR("Failed to set exposure time. Error code: {}", this->return_code_);
        return false;
    }
    SPDLOG_INFO("Exposure time set to: {} us", exposure_time);
    return true;
}

void OptCamera::disconnect() {
    if (this->is_connected_) {
        this->return_code_ = OPT_StopGrabbing(this->camera_handle_);
        if (this->return_code_ != OPT_OK) {
            SPDLOG_ERROR("Failed to stop grabbing. Error code: {}", this->return_code_);
        }
        this->is_grabbing_ = false;
        SPDLOG_INFO("Grabbing stopped.");

        this->return_code_ = OPT_Close(this->camera_handle_);
        if (this->return_code_ != OPT_OK) {
            SPDLOG_ERROR("Failed to close camera. Error code: {}", this->return_code_);
        }
        this->is_connected_ = false;
        SPDLOG_INFO("Camera disconnected.");
    }
}

OptCamera::~OptCamera() {
    this->disconnect();
}
}  // namespace camera_opt
