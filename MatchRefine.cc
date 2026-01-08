#include <iostream>
#include <opencv2/opencv.hpp>
#include <chrono>
#include <iomanip>
#include <sstream>
#include "OnnxHelper.h"
#include "XFeat.h"
#include "Matcher.h"
#include "camera_opt/include/OptCamera.h"


int main(int argc, char** argv) {
    std::string modelFile = "../../model/xfeat_640x640.onnx";
    std::string imgFile1  = ""; // no default: template must be provided via --img1 or captured from camera
    std::string imgFile2  = ""; // no default
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
            std::cout << "Usage: --model <model> --img1 <img1> [--img2 <img2>] --ransac <0|1>\n";
            std::cout << "  If both --img1 and --img2 are set: static image matching mode\n";
            std::cout << "  Otherwise: live stream matching mode (requires camera)\n";
            return 0;
        }
    }

    std::cout << "Model file: " << modelFile << std::endl;
    std::cout << "Image file 1: " << (imgFile1.empty() ? "(none - will capture from camera)" : imgFile1) << std::endl;
    std::cout << "Image file 2: " << (imgFile2.empty() ? "(none - will use live stream)" : imgFile2) << std::endl;
    std::cout << "Use RANSAC: " << (useRansac ? "true" : "false") << std::endl;

    // Determine mode: static image matching vs. live stream matching
    bool staticMode = !imgFile1.empty() && !imgFile2.empty();
    std::cout << "Mode: " << (staticMode ? "Static image matching" : "Live stream matching") << std::endl;

    // create XFeat object
    std::cout << "Creating XFeat...\n";
    try {
        XFeat xfeat(modelFile);

        // Template image: either from file (--img1) or captured from first grabbed frame ROI.
        cv::Mat templateImg;
        std::vector<cv::KeyPoint> keysT;
        cv::Mat descsT;

        // Only connect to camera for stream mode
        OptCamera camera(0);
        bool cameraReady = false;
        if (!staticMode) {
            std::cout << "Connecting to camera for stream mode..." << std::endl;
            cameraReady = camera.connect() && camera.startGrabbing();
        }

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
        // For stream mode, template must be provided or captured from camera
        if (!cameraReady) {
            std::cerr << "ERROR: No template image provided and camera not available. "
                      << "Provide --img1 or ensure camera is connected." << std::endl;
            return -1;
        }

        std::cout << "\nNo template provided. Capturing from live stream..." << std::endl;
        std::cout << "Press 'c' to capture a frame for ROI selection, 'q' to quit." << std::endl;
        
        cv::Mat liveFrame;
        int captureRetries = 300; // ~10 seconds at 30fps
        while (captureRetries-- > 0) {
            cv::Mat f = camera.captureImage(500);
            if (f.empty()) {
                int k = cv::waitKey(30);
                if (k == 'q' || k == 27) {
                    std::cout << "Capture aborted by user." << std::endl;
                    camera.stopGrabbing();
                    camera.disconnect();
                    return -1;
                }
                continue;
            }
            
            cv::Mat disp;
            if (f.channels() == 3) cv::cvtColor(f, disp, cv::COLOR_BGR2GRAY);
            else disp = f;
            cv::resize(disp, disp, cv::Size(640, 640));
            
            cv::imshow("Live Preview - Press 'c' to Capture", disp);
            int k = cv::waitKey(30);
            if (k == 'c') {
                liveFrame = disp.clone();
                std::cout << "Frame captured. Select ROI using mouse." << std::endl;
                break;
            }
            if (k == 'q' || k == 27) {
                std::cout << "Capture cancelled by user." << std::endl;
                camera.stopGrabbing();
                camera.disconnect();
                return -1;
            }
        }
        
        cv::destroyWindow("Live Preview - Press 'c' to Capture");
        
        if (liveFrame.empty()) {
            std::cerr << "ERROR: Failed to capture frame within timeout." << std::endl;
            camera.stopGrabbing();
            camera.disconnect();
            return -1;
        }

        std::cout << "Please select ROI on the captured image using mouse. Press ENTER or SPACE when done." << std::endl;
        cv::Rect2d roi = cv::selectROI("Select ROI", liveFrame);
        cv::destroyWindow("Select ROI");
        
        if (roi.width <= 0 || roi.height <= 0) {
            std::cerr << "ERROR: Invalid ROI selected." << std::endl;
            camera.stopGrabbing();
            camera.disconnect();
            return -1;
        }
        
        templateImg = liveFrame(roi).clone();
        cv::resize(templateImg, templateImg, cv::Size(640, 640));
        std::cout << "Template ROI extracted and resized to 640x640." << std::endl;
    }

    // compute features for template
    xfeat.DetectAndCompute(templateImg, keysT, descsT, 1000);

    if (keysT.empty() || descsT.empty()) {
        std::cerr << "No features found in template." << std::endl;
        return -1;
    }

    std::cout << "Template ready (" << keysT.size() << " features)." << std::endl;

    // ============== STATIC MODE: Match two provided images ==============
    if (staticMode) {
        std::cout << "Loading second image: " << imgFile2 << std::endl;
        cv::Mat img2 = cv::imread(imgFile2, cv::IMREAD_GRAYSCALE);
        if (img2.empty()) {
            std::cerr << "Failed to read second image: " << imgFile2 << std::endl;
            return -1;
        }
        cv::resize(img2, img2, cv::Size(640, 640));

        std::vector<cv::KeyPoint> keysF;
        cv::Mat descsF;
        xfeat.DetectAndCompute(img2, keysF, descsF, 1000);

        std::vector<cv::DMatch> matches;
        if (!keysF.empty() && !descsF.empty()) {
            Matcher::Match(descsT, descsF, matches, 0.82f);
            // Matcher::gridFilterMatches(keysF, matches,
            //       img2.cols, img2.rows);
        }
        std::vector<cv::Point2f> ptsT, ptsF;
        for (auto &m : matches) {
            ptsT.push_back(keysT[m.queryIdx].pt);
            ptsF.push_back(keysF[m.trainIdx].pt);
        }
        Matcher::RejectBadMatchesF(ptsT, ptsF, matches, 4.0f);
        Matcher::gridFilterMatches(keysF, matches,
                  img2.cols, img2.rows);    

        std::cout << "Matching features: template(" << keysT.size() << ") vs image2(" << keysF.size() << ")" << std::endl;

        std::cout << "Matched: " << matches.size() << " correspondences" << std::endl;

        // draw matches
        cv::Mat templColor, img2Color;
        cv::cvtColor(templateImg, templColor, cv::COLOR_GRAY2BGR);
        cv::cvtColor(img2, img2Color, cv::COLOR_GRAY2BGR);
        cv::Mat imgMatches;
        cv::drawMatches(templColor, keysT, img2Color, keysF, matches, imgMatches);

        // compute homography if possible
        double homography_conf = 0.0;
        if (matches.size() >= 4) {
            std::vector<cv::Point2f> ptsT, ptsF;
            for (auto &m : matches) {
                ptsT.push_back(keysT[m.queryIdx].pt);
                ptsF.push_back(keysF[m.trainIdx].pt);
            }
            cv::Mat inlierMask;
            // cv::Mat H = cv::findHomography(ptsT, ptsF, cv::RANSAC, 4.0, inlierMask);
            cv::Mat H = cv::findHomography(ptsT, ptsF, cv::USAC_MAGSAC, 4.0, inlierMask, 700, 0.995);
            if (!H.empty() && !inlierMask.empty()) {
                int inliers = cv::countNonZero(inlierMask);
                homography_conf = (double)inliers / (double)matches.size();

                /* ---------- non-linear refinement ---------- */
                // std::vector<cv::Point2f> inT, inF;
                // for (int i = 0; i < inlierMask.rows; ++i)
                //     if (inlierMask.at<uchar>(i)) {
                //         inT.push_back(ptsT[i]);
                //         inF.push_back(ptsF[i]);
                //     }

                // if (inT.size() >= 8)
                //     Matcher::refineHomography(H, inT, inF);

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

        std::ostringstream oss;
        oss.setf(std::ios::fixed);
        oss << std::setprecision(2) << "matches: " << matches.size() << "  H_conf: " << homography_conf;
        cv::putText(imgMatches, oss.str(), cv::Point(10, 20), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 255, 255), 2);

        cv::imshow("matches", imgMatches);
        std::cout << "Displaying result. Press any key to exit." << std::endl;
        cv::waitKey(0);
        return 0;
    }

    // ============== LIVE STREAM MODE ==============
    // Camera must be ready for live stream
    if (!cameraReady) {
        std::cerr << "ERROR: Camera not available for continuous matching." << std::endl;
        return -1;
    }

    std::cout << "\n=============== LIVE STREAM MATCHING ===============" << std::endl;
    std::cout << "Press 't' or 'c' to reselect template ROI" << std::endl;
    std::cout << "Press ESC to exit" << std::endl;
    std::cout << "===================================================\n" << std::endl;

    // for FPS calculation
    double fps = 0.0;
    auto last_ts = std::chrono::high_resolution_clock::now();
    int frameCount = 0;
    
    while (true) {
        cv::Mat frame = camera.captureImage(1000);
        if (frame.empty()) {
            int kk = cv::waitKey(1);
            if (kk == 27) break; // ESC
            continue;
        }

        frameCount++;
        auto now = std::chrono::high_resolution_clock::now();
        double dt = std::chrono::duration<double>(now - last_ts).count();
        last_ts = now;
        
        if (dt > 0.0) {
            double cur_fps = 1.0 / dt;
            fps = fps * 0.90 + cur_fps * 0.10; // exponential smoothing
        }

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
            Matcher::gridFilterMatches(keysF, matches,
                  gray.cols, gray.rows);
        }

        // draw matches (template left, frame right)
        cv::Mat templColor, frameColor;
        cv::cvtColor(templateImg, templColor, cv::COLOR_GRAY2BGR);
        cv::cvtColor(gray, frameColor, cv::COLOR_GRAY2BGR);
        cv::Mat imgMatches;
        cv::drawMatches(templColor, keysT, frameColor, keysF, matches, imgMatches);

        // compute homography from template to frame if possible
        double homography_conf = 0.0;
        int matchCount = (int)matches.size();
        if (matches.size() >= 4) {
            std::vector<cv::Point2f> ptsT, ptsF;
            for (auto &m : matches) {
                ptsT.push_back(keysT[m.queryIdx].pt);
                ptsF.push_back(keysF[m.trainIdx].pt);
            }
            cv::Mat inlierMask;
            cv::Mat H = cv::findHomography(ptsT, ptsF, cv::USAC_MAGSAC, 4.0, inlierMask, 700, 0.995);
            if (!H.empty() && !inlierMask.empty()) {
                int inliers = cv::countNonZero(inlierMask);
                homography_conf = matchCount > 0 ? (double)inliers / (double)matchCount : 0.0;

                /* ---------- non-linear refinement ---------- */
                std::vector<cv::Point2f> inT, inF;
                for (int i = 0; i < inlierMask.rows; ++i)
                    if (inlierMask.at<uchar>(i)) {
                        inT.push_back(ptsT[i]);
                        inF.push_back(ptsF[i]);
                    }

                if (inT.size() >= 8)
                    Matcher::refineHomography(H, inT, inF);

                // draw warped template corners on frame
                std::vector<cv::Point2f> cornersT = {
                    {0, 0},
                    {(float)templateImg.cols, 0},
                    {(float)templateImg.cols, (float)templateImg.rows},
                    {0, (float)templateImg.rows}
                };
                std::vector<cv::Point2f> cornersF;
                cv::perspectiveTransform(cornersT, cornersF, H);

                int offsetX = templColor.cols;
                std::vector<cv::Point> poly;
                for (auto &p : cornersF) {
                    poly.push_back(cv::Point((int)std::round(p.x) + offsetX, (int)std::round(p.y)));
                }
                const cv::Point* pts = poly.data();
                int npts = (int)poly.size();
                cv::polylines(imgMatches, &pts, &npts, 1, true, cv::Scalar(0, 255, 0), 2);
            }
        }

        // overlay FPS and match info
        std::ostringstream oss;
        oss.setf(std::ios::fixed);
        oss << std::setprecision(1) << "FPS:" << fps;
        oss << "  matches:" << matchCount;
        oss << std::setprecision(2) << "  H_conf:" << homography_conf;
        cv::putText(imgMatches, oss.str(), cv::Point(10, 20),
                   cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 255, 255), 2);

        cv::imshow("Live Stream Matching", imgMatches);

        int key = cv::waitKey(1);
        if (key == 't' || key == 'c') {
            std::cout << "\nReselecting template ROI from current frame..." << std::endl;
            cv::Rect2d roi = cv::selectROI("Select New Template ROI", gray);
            cv::destroyWindow("Select New Template ROI");
            
            if (roi.width > 0 && roi.height > 0) {
                templateImg = gray(roi).clone();
                cv::resize(templateImg, templateImg, cv::Size(640, 640));
                xfeat.DetectAndCompute(templateImg, keysT, descsT, 1000);
                std::cout << "New template set with " << keysT.size() << " features.\n" << std::endl;
            } else {
                std::cout << "ROI selection cancelled. Continuing with previous template.\n" << std::endl;
            }
            continue;
        }
        if (key == 27) break; // ESC
    }

    std::cout << "\nStream matching stopped. Processed " << frameCount << " frames." << std::endl;
    camera.stopGrabbing();
    camera.disconnect();
    }
    catch (const Ort::Exception& e) {
        std::cout << "ERROR: " << e.what() << std::endl;
    }

    return 0;
}