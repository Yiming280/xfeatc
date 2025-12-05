#include <camera_opt.h>
#include <opencv2/opencv.hpp>
#include <yaml-cpp/yaml.h>
#include <spdlog/spdlog.h>
#include <filesystem>
#include <iostream>

int main(int argc, char** argv) {
    std::string capture_params_yaml = "C:/Users/yiming.wei/source/repos/xfeatc/_configs/capture_params_opt.yaml";
    if (argc >= 2) {
        capture_params_yaml = argv[1];
    }
    YAML::Node capture_params         = YAML::LoadFile(capture_params_yaml);
    const std::string camera_key      = capture_params["camera_key"].as<std::string>("camera_0");
    const double camera_exposure_time = capture_params["camera_exposure"].as<double>(10000);
    const std::string save_directory  = capture_params["save_directory"].as<std::string>("../calibration_data");

    SPDLOG_INFO("Capture parameters:");
    SPDLOG_INFO("   Camera key: {}", camera_key);
    SPDLOG_INFO("   Camera exposure time: {} us", camera_exposure_time);
    SPDLOG_INFO("   Save directory: {}", save_directory);

    if (!std::filesystem::exists(save_directory)) {
        std::filesystem::create_directory(save_directory);
        SPDLOG_INFO("Created save directory: {}", save_directory);
    } else {
        SPDLOG_INFO("Save directory already exists: {}", save_directory);
    }

    std::time_t t = std::time(nullptr);
    std::tm tm    = *std::localtime(&t);
    std::stringstream subdirectory_name;
    subdirectory_name << save_directory << "/" << std::put_time(&tm, "%Y-%m-%d_%H-%M-%S");
    if (!std::filesystem::exists(subdirectory_name.str())) {
        std::filesystem::create_directory(subdirectory_name.str());
        SPDLOG_INFO("Created subdirectory: {}", subdirectory_name.str());
    } else {
        SPDLOG_INFO("Subdirectory already exists: {}", subdirectory_name.str());
    }

    camera_opt::OptCamera camera(camera_key);

    std::string_view window_name = "camera-feed";
    cv::namedWindow(window_name.data(), cv::WINDOW_NORMAL | cv::WINDOW_GUI_EXPANDED);

    if (!camera.connect()) {
        std::cerr << "Could not connect to the camera." << std::endl;
        return -1;
    }
    bool exposure_set = camera.setExposureTime(camera_exposure_time);
    if (!exposure_set) {
        std::cerr << "Could not set the camera exposure time." << std::endl;
        return -1;
    }

    int image_counter = 0;
    while (true) {
        cv::Mat image_from_camera = camera.captureImage();

        cv::Mat resized_camera_image;
        cv::resize(image_from_camera, resized_camera_image, cv::Size(), 0.25, 0.25);
        cv::imshow(window_name.data(), resized_camera_image);

        int key = cv::waitKey(1);
        if (key == 'c') {
            std::stringstream formatted_name;
            formatted_name << subdirectory_name.str() << "/" << std::setw(3) << std::setfill('0') << image_counter
                           << ".png";
            bool save_success = cv::imwrite(formatted_name.str(), image_from_camera);
            if (save_success) {
                SPDLOG_INFO("Saved image: {}", formatted_name.str());
            } else {
                throw std::runtime_error("Could not save image to disk.");
            }
            image_counter++;
        }

        if (key == 'q' || key == 3 || key == 27) {
            break;
        }
    }

    camera.disconnect();
    return 0;
}
