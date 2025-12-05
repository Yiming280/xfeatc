#include <iostream>
#include <opencv2/opencv.hpp>
#include "OnnxHelper.h"
#include "XFeat.h"
#include "Matcher.h"
#include "camera_opt/include/OptCamera.h"


int main(int argc, char** argv) {
    std::string modelFile = "../../model/xfeat_640x640.onnx";
    std::string imgFile1  = ""; // no default: template must be provided via --img1 or captured from camera
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

    // Template image: either from file (--img1) or captured from first grabbed frame ROI.
    cv::Mat templateImg;
    std::vector<cv::KeyPoint> keysT;
    cv::Mat descsT;

    // connect to camera for stream
    OptCamera camera(0);
    bool cameraReady = camera.connect() && camera.startGrabbing();

    // If a template path is provided, load it. Otherwise capture from live preview when user presses 'c'.
    if (!imgFile1.empty()) {
        templateImg = cv::imread(imgFile1, cv::IMREAD_GRAYSCALE);
        if (templateImg.empty()) {
            std::cerr << "Failed to read template image: " << imgFile1 << std::endl;
        } else {
            cv::resize(templateImg, templateImg, cv::Size(640, 640));
        }
    }

    if (templateImg.empty()) {
        if (!cameraReady) {
            std::cerr << "No template image provided and camera not available to select ROI." << std::endl;
            return -1;
        }

        std::cout << "No template provided. Showing live preview. Press 'c' to capture a frame for ROI selection, 'q' to quit." << std::endl;
        cv::namedWindow("Live Preview", cv::WINDOW_NORMAL);
        cv::Mat liveFrame;
        while (true) {
            cv::Mat f = camera.captureImage(500);
            if (f.empty()) {
                int k = cv::waitKey(30);
                if (k == 'q' || k == 27) {
                    std::cerr << "No frame captured. Exiting." << std::endl;
                    return -1;
                }
                continue;
            }
            cv::Mat disp;
            if (f.channels() == 3) cv::cvtColor(f, disp, cv::COLOR_BGR2GRAY);
            else disp = f;
            cv::resize(disp, disp, cv::Size(640, 640));
            cv::imshow("Live Preview", disp);
            int k = cv::waitKey(30);
            if (k == 'c') {
                liveFrame = disp.clone();
                break;
            }
            if (k == 'q' || k == 27) {
                std::cerr << "User aborted template capture." << std::endl;
                return -1;
            }
        }
        cv::destroyWindow("Live Preview");

        std::cout << "Please select ROI on the captured image. Press ENTER or SPACE when done." << std::endl;
        cv::Rect2d roi = cv::selectROI("Select ROI", liveFrame);
        cv::destroyWindow("Select ROI");
        if (roi.width <= 0 || roi.height <= 0) {
            std::cerr << "Invalid ROI selected." << std::endl;
            return -1;
        }
        templateImg = liveFrame(roi).clone();
        cv::resize(templateImg, templateImg, cv::Size(640, 640));
    }

    // compute features for template
    xfeat.DetectAndCompute(templateImg, keysT, descsT, 1000);

    if (keysT.empty() || descsT.empty()) {
        std::cerr << "No features found in template." << std::endl;
        return -1;
    }

    std::cout << "Template ready (" << keysT.size() << " features)." << std::endl;

    // Continuous matching loop
    if (!cameraReady) {
        std::cerr << "Camera not available for continuous matching." << std::endl;
        return -1;
    }

    std::cout << "Starting continuous matching. Press ESC to exit, 't' to reselect ROI." << std::endl;
    // for FPS calculation
    double fps = 0.0;
    auto last_ts = std::chrono::high_resolution_clock::now();
    while (true) {
        cv::Mat frame = camera.captureImage(1000);
        if (frame.empty()) {
            int kk = cv::waitKey(1);
            if (kk == 27) break;
            continue;
        }

        auto now = std::chrono::high_resolution_clock::now();
        double dt = std::chrono::duration<double>(now - last_ts).count();
        last_ts = now;
        double cur_fps = dt > 0.0 ? 1.0 / dt : 0.0;
        fps = fps * 0.90 + cur_fps * 0.10;

        cv::Mat gray;
        if (frame.channels() == 3) cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
        else gray = frame;
        cv::resize(gray, gray, cv::Size(640, 640));

        // detect features on live frame
        std::vector<cv::KeyPoint> keysF;
        cv::Mat descsF;
        xfeat.DetectAndCompute(gray, keysF, descsF, 1000);

        std::vector<cv::DMatch> matches;
        if (!keysF.empty() && !descsF.empty()) {
            Matcher::Match(descsT, descsF, matches, 0.82f);
            if (useRansac && !matches.empty()) {
                std::vector<cv::Point2f> ptsT, ptsF;
                for (auto &m : matches) {
                    ptsT.push_back(keysT[m.queryIdx].pt);
                    ptsF.push_back(keysF[m.trainIdx].pt);
                }
                Matcher::RejectBadMatchesF(ptsT, ptsF, matches, 4.0f);
            }
        }

        // draw matches (template left, frame right)
        cv::Mat templColor, frameColor;
        cv::cvtColor(templateImg, templColor, cv::COLOR_GRAY2BGR);
        cv::cvtColor(gray, frameColor, cv::COLOR_GRAY2BGR);
        cv::Mat imgMatches;
        cv::drawMatches(templColor, keysT, frameColor, keysF, matches, imgMatches);

        // compute homography from template to frame if possible, also compute inlier ratio
        double homography_conf = 0.0;
        int matchCount = (int)matches.size();
        if (matches.size() >= 4) {
            std::vector<cv::Point2f> ptsT, ptsF;
            for (auto &m : matches) {
                ptsT.push_back(keysT[m.queryIdx].pt);
                ptsF.push_back(keysF[m.trainIdx].pt);
            }
            cv::Mat inlierMask;
            cv::Mat H = cv::findHomography(ptsT, ptsF, cv::RANSAC, 4.0, inlierMask);
            if (!H.empty() && !inlierMask.empty()) {
                int inliers = cv::countNonZero(inlierMask);
                homography_conf = matchCount > 0 ? (double)inliers / (double)matchCount : 0.0;

                std::vector<cv::Point2f> cornersT = { {0,0}, { (float)templateImg.cols, 0 }, { (float)templateImg.cols, (float)templateImg.rows }, { 0, (float)templateImg.rows } };
                std::vector<cv::Point2f> cornersF;
                cv::perspectiveTransform(cornersT, cornersF, H);

                int offsetX = templColor.cols;
                std::vector<cv::Point> poly;
                for (auto &p : cornersF) poly.push_back(cv::Point((int)std::round(p.x) + offsetX, (int)std::round(p.y)));
                const cv::Point* pts = poly.data();
                int npts = (int)poly.size();
                cv::polylines(imgMatches, &pts, &npts, 1, true, cv::Scalar(0,255,0), 2);
            }
        }

        // overlay FPS and match info
        std::ostringstream oss;
        oss.setf(std::ios::fixed); oss<<std::setprecision(1);
        oss<<"FPS:"<<fps<<"  matches:"<<matchCount<<"  H_conf:"<<std::setprecision(2)<<homography_conf;
        cv::putText(imgMatches, oss.str(), cv::Point(10,20), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0,255,255), 2);

        cv::imshow("matches", imgMatches);

        int key = cv::waitKey(1);
        if (key == 't' || key == 'c') {
            std::cout << "Reselecting ROI..." << std::endl;
            cv::Rect2d roi = cv::selectROI("Select ROI", gray);
            cv::destroyWindow("Select ROI");
            if (roi.width > 0 && roi.height > 0) {
                templateImg = gray(roi).clone();
                cv::resize(templateImg, templateImg, cv::Size(640, 640));
                xfeat.DetectAndCompute(templateImg, keysT, descsT, 1000);
                std::cout << "New template features: " << keysT.size() << std::endl;
            }
            continue;
        }
        if (key == 27) break;
    }

    camera.stopGrabbing();
    camera.disconnect();
    }
    catch (const Ort::Exception& e) {
        std::cout << "ERROR: " << e.what() << std::endl;
    }

    return 0;
}