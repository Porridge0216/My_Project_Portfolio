//
// Created by Porridge on 2023/9/9.
//
#include "generalFunctions.h"

cv::Mat normalize(const cv::Mat &points) {
    // Accept homogeneous coordinate points dim*N
    // Return coordinates in N*(dim-1)
    if (points.rows == 4) {
        // 3D
        cv::Mat last_row_mat;
        cv::vconcat(points.row(3), points.row(3), last_row_mat);
        cv::vconcat(last_row_mat, points.row(3), last_row_mat);
        cv::Mat pointsT = points.rowRange(0, 3) / last_row_mat;
        return pointsT.t();
    } else if (points.rows == 3) {
        // 2D
        cv::Mat last_row_mat;
        cv::vconcat(points.row(2), points.row(2), last_row_mat);
        cv::Mat pointsT = points.rowRange(0, 2) / last_row_mat;
        return pointsT.t();
    } else {
        throw std::runtime_error("NormalizeError");
    }

}

double slope(const cv::Point2d &point1, const cv::Point2d &point2) {
    cv::Point2d vec = point2 - point1;
    return vec.y / vec.x;
}

double distance(const cv::Point2d &point1, const cv::Point2d &point2) {
    return cv::norm(point2 - point1);
}

cv::Point2d findPointInLineByX(const cv::Point2d &point1, const cv::Point2d &point2, const double &x) {
    double y = point1.y + (x - point1.x) * (point2.y - point1.y) / (point2.x - point1.x);
    return {x, y};
}

cv::Point2d findPointInLineByY(const cv::Point2d &point1, const cv::Point2d &point2, const double &y) {
    double x = point1.x + (y - point1.y) * (point2.x - point1.x) / (point2.y - point1.y);
    return {x, y};
}

cv::Point2d findPointInLineByAspectRatio(const cv::Point2d &point1, const cv::Point2d &point2,
                                         const cv::Point2d &launch_point, const double &aspect_ratio) {
    double k1 = slope(point2, point1);
    double k2 = -1 / aspect_ratio;
    double b1 = point1.y - k1 * point1.x;
    double b2 = launch_point.y - k2 * launch_point.x;
    double x = (b2 - b1) / (k1 - k2);
    double y = (k1 * b2 - k2 * b1) / (k1 - k2);
    return {x, y};
}

cv::Point2d findIntersection(const cv::Point2d &point1, const double &k1, const cv::Point2d &point2, const double &k2) {
    double x = (k1 * point1.x - k2 * point2.x + point2.y - point1.y) / (k1 - k2);
    double y = k1 * x + point1.y - k1 * point1.x;
    return {x, y};
}

cv::Mat shrink(const cv::Mat &points) {
    std::vector<cv::Point2d> points_v;
    for (int i = 0; i < points.rows; ++i) {
        double x = points.at<double>(i, 0);
        double y = points.at<double>(i, 1);
        points_v.emplace_back(x, -y);
    }
    cv::Point2d A = points_v[0];
    cv::Point2d B = points_v[1];
    cv::Point2d C = points_v[2];
    cv::Point2d D = points_v[3];
    double w = std::abs(B.x - A.x);
    double h = std::abs(A.y - D.y);
    cv::Point2d center = 0.25 * (A + B + C + D);
    std::vector<cv::Point2d> rect_shrunken_v;
    rect_shrunken_v.emplace_back(center.x - 0.49 * w, center.y - 0.49 * h);
    rect_shrunken_v.emplace_back(center.x + 0.49 * w, center.y - 0.49 * h);
    rect_shrunken_v.emplace_back(center.x + 0.49 * w, center.y + 0.49 * h);
    rect_shrunken_v.emplace_back(center.x - 0.49 * w, center.y + 0.49 * h);
    cv::Mat rect_shrunken = (cv::Mat_<double>(4, 2)
            <<
            rect_shrunken_v[0].x, rect_shrunken_v[0].y,
            rect_shrunken_v[1].x, rect_shrunken_v[1].y,
            rect_shrunken_v[2].x, rect_shrunken_v[2].y,
            rect_shrunken_v[3].x, rect_shrunken_v[3].y);
    return rect_shrunken;
}

