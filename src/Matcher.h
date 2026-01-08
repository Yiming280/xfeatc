#pragma once

#include <iostream>
#include <opencv2/opencv.hpp>


class Matcher {
public:

    static void Match(const cv::Mat &descs1, const cv::Mat &descs2, std::vector<cv::DMatch> &matches, float minScore = 0.82f);

    static bool RejectBadMatchesF(std::vector<cv::Point2f> &pts1,
                                  std::vector<cv::Point2f> &pts2,
                                  std::vector<cv::DMatch> &matches,
                                  float thresh);

    static void gridFilterMatches(const std::vector<cv::KeyPoint>& kps, std::vector<cv::DMatch>& matches, int img_w, int img_h, int gx = 8, int gy = 8,
        int max_per_cell = 5);

    static cv::Mat reprojectionError(const cv::Mat& H, const std::vector<cv::Point2f>& ptsT, const std::vector<cv::Point2f>& ptsF);

    static cv::Mat numericalJacobian(const cv::Mat& H, const std::vector<cv::Point2f>& ptsT, const std::vector<cv::Point2f>& ptsF, double eps);

    static void refineHomography(cv::Mat& H, const std::vector<cv::Point2f>& ptsT, const std::vector<cv::Point2f>& ptsF, int iterations=5);


};