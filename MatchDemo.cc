#include <iostream>
#include <opencv2/opencv.hpp>
#include "OnnxHelper.h"
#include "XFeat.h"
#include "Matcher.h"


int main(int argc, char** argv) {
    std::string modelFile = "../../model/xfeat_640x640.onnx";
    std::string imgFile1  = "../../data/1.png";
    std::string imgFile2  = "../../data/2.png";
    int useRansac = 1;

    // 手动解析命令行参数
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--model" && i + 1 < argc) {
            modelFile = argv[++i];
        } else if (arg == "--img1" && i + 1 < argc) {
            imgFile1 = argv[++i];
        } else if (arg == "--img2" && i + 1 < argc) {
            imgFile2 = argv[++i];
        } else if (arg == "--ransac" && i + 1 < argc) {
            useRansac = std::stoi(argv[++i]);
        } else if (arg == "--help") {
            std::cout << "Usage: --model <model> --img1 <img1> --img2 <img2> --ransac <0|1>\n";
            return 0;
        }
    }

    std::cout << "Model file: " << modelFile << std::endl;
    std::cout << "Image file 1: " << imgFile1 << std::endl;
    std::cout << "Image file 2: " << imgFile2 << std::endl;
    std::cout << "Use RANSAC: " << (useRansac ? "true" : "false") << std::endl;

    // create XFeat object
    std::cout << "creating XFeat...\n";
    try {
    XFeat xfeat(modelFile);

    // read images
    std::cout << "reading images...\n";
    cv::Mat img1 = cv::imread(imgFile1, cv::IMREAD_GRAYSCALE);
    cv::Mat img2 = cv::imread(imgFile2, cv::IMREAD_GRAYSCALE);
    if (img1.empty()) {
        std::cerr << "Failed to read img1: " << imgFile1 << std::endl;
        return -1;
    }
    if (img2.empty()) {
        std::cerr << "Failed to read img2: " << imgFile2 << std::endl;
        return -1;
    }
    cv::resize(img1, img1, cv::Size(640, 640));
    cv::resize(img2, img2, cv::Size(640, 640));
    // detect xfeat corners and compute descriptors
    std::cout << "detecting features ...\n";
    std::vector<cv::KeyPoint> keys1, keys2;
    cv::Mat descs1, descs2;
    xfeat.DetectAndCompute(img1, keys1, descs1, 1000);
    xfeat.DetectAndCompute(img2, keys2, descs2, 1000);

    // draw keypoints
    cv::Mat imgColor1, imgColor2;
    cv::cvtColor(img1, imgColor1, cv::COLOR_GRAY2BGR);
    cv::cvtColor(img2, imgColor2, cv::COLOR_GRAY2BGR);
    cv::drawKeypoints(imgColor1, keys1, imgColor1, cv::Scalar(0, 0, 255));
    cv::drawKeypoints(imgColor2, keys2, imgColor2, cv::Scalar(0, 0, 255));
    cv::putText(imgColor1, "features: " + std::to_string(keys1.size()), cv::Point(5, 30), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 255, 255), 1);
    cv::putText(imgColor2, "features: " + std::to_string(keys2.size()), cv::Point(5, 30), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 255, 255), 1);
    cv::imshow("image1", imgColor1);
    cv::imshow("image2", imgColor2);
    cv::waitKey(0);

    // matching
    std::cout << "matching ...\n";
    std::vector<cv::DMatch> matches;
    Matcher::Match(descs1, descs2, matches, 0.82f);

    // remove outlier matches, notice that this only works for undistorted keypoints.
    // If you have distorted keypoints (fisheye images for example), you need to undistort them first.
    if (useRansac) {
        std::vector<cv::Point2f> pts1, pts2;
        for (auto& m : matches) {
            pts1.push_back(keys1[m.queryIdx].pt);
            pts2.push_back(keys2[m.trainIdx].pt);
        }
        Matcher::RejectBadMatchesF(pts1, pts2, matches, 4.0f);
    }

    // draw matches
    cv::Mat imgMatches;
    cv::drawMatches(imgColor1, keys1, imgColor2, keys2, matches, imgMatches);
    cv::imshow("matches", imgMatches);
    cv::waitKey(0);
    }
    catch (const Ort::Exception& e) {
        std::cout << "ERROR: " << e.what() << std::endl;
    }

    return 0;
}