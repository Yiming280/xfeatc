#include <iostream>
#include <opencv2/opencv.hpp>
#include <chrono>
#include <iomanip>
#include "OnnxHelper.h"
#include "XFeat.h"
#include "camera_opt/include/OptCamera.h"


int main(int argc, char** argv) {
    // Default paths - no default image file
    std::string modelFile = "../../model/xfeat_640x640.onnx";
    std::string imgFile   = "";

    // Parse command line arguments
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--model" && i + 1 < argc) {
            modelFile = argv[++i];
        } else if (arg == "--img" && i + 1 < argc) {
            imgFile = argv[++i];
        } else if (arg == "--help") {
            std::cout << "Usage: DetectDemo [--model <model_path>] [--img <image_path>]\n";
            std::cout << "  If --img provided: static image detection (no camera needed)\n";
            std::cout << "  If --img not provided: live stream detection (camera required)\n";
            std::cout << "  Press ESC to exit\n";
            return 0;
        }
    }

    std::cout << "========================================" << std::endl;
    std::cout << "XFeat Feature Detection Demo" << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << "Model file: " << modelFile << std::endl;
    
    bool staticMode = !imgFile.empty();
    if (staticMode) {
        std::cout << "Image file: " << imgFile << std::endl;
        std::cout << "Mode: Static image detection" << std::endl;
    } else {
        std::cout << "Mode: Live stream detection" << std::endl;
    }

    try {
        XFeat xfeat(modelFile);

        // ============== STATIC MODE ==============
        if (staticMode) {
            cv::Mat img = cv::imread(imgFile, cv::IMREAD_GRAYSCALE);
            if (img.empty()) {
                std::cerr << "ERROR: Failed to load image: " << imgFile << std::endl;
                return -1;
            }

            cv::resize(img, img, cv::Size(640, 640));
            std::cout << "Detecting features..." << std::endl;

            std::vector<cv::KeyPoint> keys;
            cv::Mat descs;
            xfeat.DetectAndCompute(img, keys, descs, 1000);

            std::cout << "Detected " << keys.size() << " features." << std::endl;

            // Draw keypoints on color image
            cv::Mat imgColor;
            cv::cvtColor(img, imgColor, cv::COLOR_GRAY2BGR);
            cv::drawKeypoints(imgColor, keys, imgColor, cv::Scalar(0, 0, 255));

            // Add feature count overlay
            std::string featureText = "Features: " + std::to_string(keys.size());
            cv::putText(imgColor, featureText, cv::Point(10, 30),
                       cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 255, 255), 2);

            cv::imshow("XFeat Detection - Static Image", imgColor);
            std::cout << "Displaying result. Press any key to exit." << std::endl;
            cv::waitKey(0);
            return 0;
        }

        // ============== LIVE STREAM MODE ==============
        std::cout << "Connecting to camera..." << std::endl;
        OptCamera camera(0);
        if (!camera.connect() || !camera.startGrabbing()) {
            std::cerr << "ERROR: Camera connection failed. "
                      << "Provide --img <path> for static image detection." << std::endl;
            return -1;
        }

        std::cout << "Camera connected. Starting detection. Press ESC to exit." << std::endl;

        double fps = 0.0;
        auto last_ts = std::chrono::high_resolution_clock::now();
        int frameCount = 0;

        while (true) {
            cv::Mat frame = camera.captureImage(1000);
            if (frame.empty()) {
                if (cv::waitKey(1) == 27) break;
                continue;
            }

            frameCount++;
            auto now = std::chrono::high_resolution_clock::now();
            double dt = std::chrono::duration<double>(now - last_ts).count();
            last_ts = now;

            if (dt > 0.0) {
                fps = fps * 0.85 + (1.0 / dt) * 0.15;
            }

            // Convert and resize
            cv::Mat gray;
            if (frame.channels() == 3) cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
            else gray = frame;
            cv::resize(gray, gray, cv::Size(640, 640));

            // Detect features
            std::vector<cv::KeyPoint> keys;
            cv::Mat descs;
            xfeat.DetectAndCompute(gray, keys, descs, 1000);

            // Draw
            cv::Mat imgColor;
            cv::cvtColor(gray, imgColor, cv::COLOR_GRAY2BGR);
            cv::drawKeypoints(imgColor, keys, imgColor, cv::Scalar(0, 0, 255));

            // Overlay metrics
            std::ostringstream oss;
            oss << std::fixed << std::setprecision(1) << "FPS: " << fps;
            cv::putText(imgColor, oss.str(), cv::Point(10, 25),
                       cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 255), 2);

            std::string featureText = "Features: " + std::to_string(keys.size());
            cv::putText(imgColor, featureText, cv::Point(10, 55),
                       cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 255), 2);

            cv::imshow("XFeat Detection - Live Stream", imgColor);

            if (cv::waitKey(1) == 27) break; // ESC
        }

        std::cout << "Stream stopped. Processed " << frameCount << " frames." << std::endl;
        camera.stopGrabbing();
        camera.disconnect();
    }
    catch (const Ort::Exception& e) {
        std::cerr << "ONNX Runtime ERROR: " << e.what() << std::endl;
        return -1;
    }

    return 0;
}