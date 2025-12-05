#include <iostream>
#include <opencv2/opencv.hpp>
#include "OnnxHelper.h"
#include "XFeat.h"
// #include <camera_opt.h>


int main(int argc, char** argv) {
    // // ------- connect camera --------
    // const std::string camera_key = "camera_0";
    // camera_opt::OptCamera camera(camera_key);
    // std::string_view window_name = "camera-feed";
    // cv::namedWindow(window_name.data(), cv::WINDOW_NORMAL | cv::WINDOW_GUI_EXPANDED);

    // if (!camera.connect()) {
    //     std::cerr << "Could not connect to the camera." << std::endl;
    //     return -1;
    // }
    // bool exposure_set = camera.setExposureTime(40000);
    // if (!exposure_set) {
    //     std::cerr << "Could not set the camera exposure time." << std::endl;
    //     return -1;
    // }
    // int image_counter = 0;
    // while (true) {
    //     cv::Mat image_from_camera = camera.captureImage();

    //     cv::Mat resized_camera_image;
    //     cv::resize(image_from_camera, resized_camera_image, cv::Size(), 0.25, 0.25);
    //     cv::imshow(window_name.data(), resized_camera_image);
    // }

    // parse arguments
    // 默认路径
    std::string modelFile = "C:/Users/yiming.wei/source/repos/xfeatc/model/xfeat_640x640.onnx";
    std::string imgFile   = "C:/Users/yiming.wei/source/repos/xfeatc/data/1.png";
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

        // read image
        std::cout << "reading image.../n";
        cv::Mat img = cv::imread(imgFile, cv::IMREAD_GRAYSCALE);
        cv::resize(img, img, cv::Size(640, 640));

        // detect xfeat corners and compute descriptors
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
    catch (const Ort::Exception& e) {
        std::cout << "ERROR: " << e.what() << std::endl;
    }

    return 0;
}