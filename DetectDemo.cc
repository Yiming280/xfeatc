#include <iostream>
#include <opencv2/opencv.hpp>
#include "OnnxHelper.h"
#include "XFeat.h"
#include "camera_opt/include/OptCamera.h"


int main(int argc, char** argv) {
    // 默认路径
    std::string modelFile = "../../model/xfeat_640x640.onnx";
    std::string imgFile   = "../../data/1.png";
    // 解析命令行参数
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--model" && i + 1 < argc) {
            modelFile = argv[++i]; // 使用下一个参数作为 model 文件
        } else if (arg == "--img" && i + 1 < argc) {
            imgFile = argv[++i];   // 使用下一个参数作为图片文件
        } else {
            std::cerr << "未知参数或缺少值: " << arg << std::endl;
            return 1;
        }
    }
    std::cout << "model file: " << modelFile << std::endl;
    std::cout << "image file: " << imgFile << std::endl;

    // create XFeat object
    std::cout << "creating XFeat.../n";
    try {
        XFeat xfeat(modelFile);
        std::cout << "Connecting to camera..." << std::endl;
        // Try continuous capture from camera. If camera not available, fallback to single image from disk.
        OptCamera camera(0); // use first enumerated camera by index
        if (camera.connect() && camera.startGrabbing()) {
            std::cout << "Camera connected. Starting continuous grab. Press ESC to exit." << std::endl;

            while (true) {
                cv::Mat frame = camera.captureImage(1000);
                if (frame.empty()) {
                    // no frame this iteration; continue
                    if (cv::waitKey(1) == 27) break; // ESC
                    continue;
                }

                // convert to single-channel grayscale expected by XFeat
                cv::Mat gray;
                if (frame.channels() == 3) cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
                else gray = frame;

                // resize to model input size
                cv::Mat input;
                cv::resize(gray, input, cv::Size(640, 640));

                // detect and compute per-frame
                std::vector<cv::KeyPoint> keys;
                cv::Mat descs;
                xfeat.DetectAndCompute(input, keys, descs, 1000);

                // draw keypoints and show
                cv::Mat imgColor;
                cv::cvtColor(input, imgColor, cv::COLOR_GRAY2BGR);
                cv::drawKeypoints(imgColor, keys, imgColor, cv::Scalar(0, 0, 255));
                cv::putText(imgColor, "features: " + std::to_string(keys.size()), cv::Point(5, 30), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 255, 255), 1);
                cv::imshow("image", imgColor);

                int k = cv::waitKey(1);
                if (k == 27) break; // ESC
            }

            camera.stopGrabbing();
            camera.disconnect();
        }
        else {
            // fallback to disk image if camera failed
            cv::Mat img = cv::imread(imgFile, cv::IMREAD_GRAYSCALE);
            if (img.empty()) {
                std::cerr << "Failed to obtain image from camera and disk: " << imgFile << std::endl;
                return -1;
            }
            cv::resize(img, img, cv::Size(640, 640));

            // single-frame detect
            std::vector<cv::KeyPoint> keys;
            cv::Mat descs;
            xfeat.DetectAndCompute(img, keys, descs, 1000);

            // draw keypoints
            cv::Mat imgColor;
            cv::cvtColor(img, imgColor, cv::COLOR_GRAY2BGR);
            cv::drawKeypoints(imgColor, keys, imgColor, cv::Scalar(0, 0, 255));
            cv::putText(imgColor, "features: " + std::to_string(keys.size()), cv::Point(5, 30), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 255, 255), 1);
            cv::imshow("image", imgColor);
            cv::waitKey(0);
        }
    }
    catch (const Ort::Exception& e) {
        std::cout << "ERROR: " << e.what() << std::endl;
    }

    return 0;
}