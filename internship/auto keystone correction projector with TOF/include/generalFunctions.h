//
// Created by Porridge on 2023/9/9.
//

#ifndef PROJECTOR_CPP_GENERALFUNCTIONS_H
#define PROJECTOR_CPP_GENERALFUNCTIONS_H

#include "opencv2/opencv.hpp"

cv::Mat normalize(const cv::Mat &points);  // Normalize coordinates, dim*N input N*(dim-1) output
double slope(const cv::Point2d &point1, const cv::Point2d &point2);  // Calculate slope between two points
double distance(const cv::Point2d &point1, const cv::Point2d &point2);  // Calculate distance between two points
// Given x, find the coordinates of a point on a line
cv::Point2d findPointInLineByX(const cv::Point2d &point1, const cv::Point2d &point2, const double &x);

// Given y, find the coordinates of a point on a line
cv::Point2d findPointInLineByY(const cv::Point2d &point1, const cv::Point2d &point2, const double &y);

// Given aspect ratio and launch point, find the coordinates of a point on a line
cv::Point2d findPointInLineByAspectRatio(const cv::Point2d &point1, const cv::Point2d &point2,
                                         const cv::Point2d &launch_point, const double &aspect_ratio);

// Find intersection of two lines
cv::Point2d findIntersection(const cv::Point2d &point1, const double &k1, const cv::Point2d &point2, const double &k2);

// Calculate intersection of a 3D vec and a 3D plane;
cv::Vec3d findIntersection(const cv::Vec3d &vec, const std::vector<double> &plane);

// Make a rectangle become a little bit smaller
cv::Mat shrink(const cv::Mat &points);





#endif //PROJECTOR_CPP_GENERALFUNCTIONS_H
