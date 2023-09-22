//
// Created by Porridge on 2023/9/16.
//

#define PI 3.1415926

#include "calibrationTOF.h"


calibrationTOF::calibrationTOF(const cv::Mat &tof_data_, const double &fov_horizontal_, const double &fov_vertical_) {
    tof_data = tof_data_;
    fov_horizontal = fov_horizontal_;
    fov_vertical = fov_vertical_;
}

void calibrationTOF::calibrate(cv::Vec3d &rotation_vector_tof2projector_) {
    calculate_xyz();
    fit_plane();
    calculate_rotation_vector();
    rotation_vector_tof2projector_ = rotation_vector_tof2projector;
}

void calibrationTOF::calculate_xyz() {
    // Calculate points_xyz
    double fov_horizontal_rad = fov_horizontal / 180 * PI;
    double fov_vertical_rad = fov_vertical / 180 * PI;
    std::vector<cv::Vec3d> points_xyz_temp;
    for (int i = 1; i <= 6; i++) {
        for (int j = 2; j <= 5; j++) {
            int depth = tof_data.at<int>(i, j);
            if (depth != -1) {
                cv::Vec3d light((j - 3.5) / 3.5 * tan(fov_horizontal_rad / 2),
                                (i - 3.5) / 3.5 * tan(fov_vertical_rad / 2),
                                1);
                light *= depth / light[2];
                points_xyz_temp.emplace_back(light);
            }
        }
    }
    // Convert to cv::Mat
    points_xyz = cv::Mat::zeros(cv::Size(3, int(points_xyz_temp.size())), CV_64FC1);
    for (int i = 0; i < points_xyz_temp.size(); i++) {
        for (int j = 0; j < 3; j++) {
            points_xyz.at<double>(i, j) = points_xyz_temp[i][j];
        }
    }
}

void calibrationTOF::fit_plane() {
    // Use points_xyz to fit a plane
    // points_xyz is a CV_64F cv::Mat, N*3
    // ax + by + c = z
    // A = [x, y, 1]
    // X = [a, b, c].T
    // B = [z]
    // Solve Ax = B
    // Construct A
    int N = points_xyz.rows;
    cv::Mat A;
    cv::Rect roi_xy(0, 0, 2, N);
    cv::Rect roi_z(2, 0, 1, N);
    cv::Mat ones = cv::Mat::ones(N, 1, CV_64F);
    cv::hconcat(points_xyz(roi_xy), ones, A);
    // Construct B
    cv::Mat B = points_xyz(roi_z);
    // Fit
    cv::solve(A, B, plane_coefficients_tof, cv::DECOMP_SVD);
}

void calibrationTOF::calculate_rotation_vector() {
    // Normal vector of plane
    cv::Vec3d n_plane(plane_coefficients_tof[0], plane_coefficients_tof[1], -1);
    double n_norm = cv::norm(n_plane); // Magnitude of normal vector
    // Wall's normal vector in wall frame
    cv::Vec3d n_wall(0.0, 0.0, -n_norm);
    // Calculate rotation vector by vector's cross product
    cv::Vec3d N_cross = n_plane.cross(n_wall);
    double theta = std::asin(cv::norm(N_cross) / std::pow(n_norm, 2));
    rotation_vector_tof2projector = N_cross * (theta / cv::norm(N_cross));
}
