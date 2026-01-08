//
// Created by d on 2024/7/13.
//

#include "Matcher.h"


template <typename T>
static void ReduceVector(std::vector<T>& v, std::vector<uchar>& status) {
    int j = 0;
    for (int i=0; i<(int)(v.size()); i++) {
        if (status[i]) {
            v[j++] = v[i];
        }
    }
    v.resize(j);
}


void Matcher::Match(const cv::Mat &descs1, const cv::Mat &descs2,
                    std::vector<cv::DMatch> &matches, float minScore) {
    cv::Mat scores12 = descs1 * descs2.t();
    cv::Mat scores21 = descs2 * descs1.t();
    std::vector<int> match12(descs1.rows, -1);
    for (int i = 0; i < scores12.rows; i++) {
        auto *row = scores12.ptr<float>(i);
        float maxScore = row[0];
        int maxIdx = 0;
        for (int j = 1; j < scores12.cols; j++) {
            if (row[j] > maxScore) {
                maxScore = row[j];
                maxIdx = j;
            }
        }
        match12[i] = maxIdx;
    }

    std::vector<int> match21(descs2.rows, -1);
    for (int i = 0; i < scores21.rows; i++) {
        auto *row = scores21.ptr<float>(i);
        float maxScore = row[0];
        int maxIdx = 0;
        for (int j = 1; j < scores21.cols; j++) {
            if (row[j] > maxScore) {
                maxScore = row[j];
                maxIdx = j;
            }
        }
        match21[i] = maxIdx;
    }

    // cross-check
    matches.clear();
    for (int i = 0; i < descs1.rows; i++) {
        int j = match12[i];
        if (match21[j] == i && scores12.at<float>(i, j) > minScore) {
            matches.emplace_back(i, j, scores12.at<float>(i, j));
        }
    }
}


bool Matcher::RejectBadMatchesF(std::vector<cv::Point2f> &pts1, std::vector<cv::Point2f> &pts2,
                                std::vector<cv::DMatch> &matches, float thresh) {
    assert(pts1.size()==pts2.size() && pts1.size()==matches.size());
    if (pts1.size() < 8) {
        return false;
    }

    std::vector<uchar> status;
    cv::findFundamentalMat(pts1, pts2, cv::FM_RANSAC, thresh, 0.999, status);
    ReduceVector(matches, status);
    return true;
}

void Matcher::gridFilterMatches(
	const std::vector<cv::KeyPoint>& kps,
	std::vector<cv::DMatch>& matches,
	int img_w, int img_h,
	int gx, int gy,
	int max_per_cell)
{
	std::vector<std::vector<std::vector<cv::DMatch>>> grid(
		gy, std::vector<std::vector<cv::DMatch>>(gx));

	for (const auto& m : matches) {
		const cv::Point2f& p = kps[m.trainIdx].pt;
		int cx = std::min(gx - 1, int(p.x / img_w * gx));
		int cy = std::min(gy - 1, int(p.y / img_h * gy));
		grid[cy][cx].push_back(m);
	}

	matches.clear();
	for (auto& row : grid)
		for (auto& cell : row) {
			std::sort(cell.begin(), cell.end(),
				[](const cv::DMatch& a, const cv::DMatch& b) {
					return a.distance < b.distance;
				});
			for (int i = 0; i < std::min((int)cell.size(), max_per_cell); ++i)
				matches.push_back(cell[i]);
		}
}

cv::Mat Matcher::reprojectionError(
	const cv::Mat& H,
	const std::vector<cv::Point2f>& ptsT,
	const std::vector<cv::Point2f>& ptsF)
{
	cv::Mat err(ptsT.size() * 2, 1, CV_64F);

	for (size_t i = 0; i < ptsT.size(); ++i) {
		cv::Mat pt = (cv::Mat_<double>(3, 1)
			<< ptsT[i].x, ptsT[i].y, 1.0);
		cv::Mat proj = H * pt;
		double x = proj.at<double>(0) / proj.at<double>(2);
		double y = proj.at<double>(1) / proj.at<double>(2);

		err.at<double>(2 * i) = x - ptsF[i].x;
		err.at<double>(2 * i + 1) = y - ptsF[i].y;
	}
	return err;
}

cv::Mat Matcher::numericalJacobian(
	const cv::Mat& H,
	const std::vector<cv::Point2f>& ptsT,
	const std::vector<cv::Point2f>& ptsF,
	double eps = 1e-6)
{
	cv::Mat J(ptsT.size() * 2, 9, CV_64F);
	cv::Mat baseErr = reprojectionError(H, ptsT, ptsF);

	for (int k = 0; k < 9; ++k) {
		cv::Mat H_eps = H.clone();
		H_eps.at<double>(k / 3, k % 3) += eps;
		cv::Mat err = reprojectionError(H_eps, ptsT, ptsF);
		cv::subtract(err, baseErr, J.col(k));
		J.col(k) /= eps;
	}
	return J;
}

void Matcher::refineHomography(
	cv::Mat& H,
	const std::vector<cv::Point2f>& ptsT,
	const std::vector<cv::Point2f>& ptsF,
	int iterations)
{
	H.convertTo(H, CV_64F);

	for (int i = 0; i < iterations; ++i) {
		cv::Mat err = reprojectionError(H, ptsT, ptsF);
		cv::Mat J = numericalJacobian(H, ptsT, ptsF);

		cv::Mat delta;
		cv::solve(J, -err, delta, cv::DECOMP_SVD);

		// 限制单步更新幅度，避免数值不稳定
		double delta_norm = cv::norm(delta);
		const double max_step = 1e-1;
		if (delta_norm > max_step) {
			delta *= (max_step / delta_norm);
		}

		for (int k = 0; k < 9; ++k)
			H.at<double>(k / 3, k % 3) += delta.at<double>(k);

		// 归一化 homography（消除尺度漂移）
		if (std::abs(H.at<double>(2,2)) > 1e-12)
			H /= H.at<double>(2,2);

		if (cv::norm(delta) < 1e-6)
			break;
	}
}