#pragma once
#include "OPTApi.h"
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <spdlog/spdlog.h>
#include <iostream>

namespace camera_opt {

/**
 * @class OptCamera
 * @brief Manages an OPT camera, including connection, image acquisition, and configuration.
 */
class OptCamera {
public:
    /**
     * @brief Constructor for the OptCamera class.
     * @param camera_key The unique key or identifier for the camera.
     */
    OptCamera(const std::string_view camera_key);

    /**
     * @brief Destructor for the OptCamera class. Ensures proper cleanup of resources.
     */
    ~OptCamera();

    /**
     * @brief Establishes a connection to the camera.
     * @return true if the connection is successful, false otherwise.
     */
    bool connect();

    /**
     * @brief Starts grabbing images from the camera.
     * @return true if the operation is successful, false otherwise.
     */
    bool startGrabbing();

    /**
     * @brief Disconnects from the camera and releases any allocated resources.
     */
    void disconnect();

    /**
     * @brief Captures a single image from the camera.
     * @return `cv::Mat` containing the captured image. Returns an empty `cv::Mat` if capture fails.
     */
    cv::Mat captureImage();

    /**
     * @brief Sets the camera's exposure time.
     * @param exposure_time The desired exposure time in microseconds.
     * @return true if the exposure time is successfully set, false otherwise.
     */
    bool setExposureTime(const double exposure_time);

    /**
     * @brief Enumerates all available camera devices.
     */
    void enumerateDevices();

private:
    /**
     * @brief The camera handle used for interacting with the OPT camera.
     */
    OPT_HANDLE camera_handle_;

    /**
     * @brief Holds the current frame data from the camera.
     */
    OPT_Frame frame_;

    /**
     * @brief Contains information about the connected camera device.
     */
    OPT_DeviceInfo* device_info_;

    /**
     * @brief Holds the list of available OPT devices.
     */
    OPT_DeviceList device_list_;

    /**
     * @brief Indicates whether the camera is currently connected.
     */
    bool is_connected_;

    /**
     * @brief Indicates whether the camera is currently grabbing images.
     */
    bool is_grabbing_;

    /**
     * @brief Stores the return code from OPT API calls.
     */
    int return_code_;

    /**
     * @brief The unique key or identifier for the camera.
     */
    std::string_view camera_key_;
};
}  // namespace camera_opt
