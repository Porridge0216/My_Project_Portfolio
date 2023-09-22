//
// Created by Porridge on 2023/9/16.
//

#define PI 3.1415926

#include "TOFCorrection.h"
#include "generalFunctions.h"
#include <iostream>

TOFCorrection::TOFCorrection(const cv::Mat &tof_data_, const cv::Vec3d &rotation_vector_tof2projector_,
                             const cv::Vec3d &gravity_standard_, const cv::Vec3d &gravity_using_,
                             const double &EOF_angle_, const double &throw_ratio_, bool biggest_,
                             const std::vector<int> &projector_resolution,
                             const double &fov_horizontal_tof_, const double &fov_vertical_tof_) {
    tof_data = tof_data_;  // TOF sensor's data when use
    rotation_vector_tof2projector = rotation_vector_tof2projector_;  // Rotation vector from TOF sensor to projector, from calibration
    gravity_standard = gravity_standard_;  // Standard gravity sensor data from calibration
    gravity_using = gravity_using_;  // Gravity sensor data when use
    EOF_angle = EOF_angle_;  // EOF angle, up, +
    throw_ratio = throw_ratio_;  // Throw ratio
    biggest = biggest_;  // Biggest or clearest
    resolution_p = projector_resolution;  // Projector resolution
    aspect_ratio = static_cast<double>(projector_resolution[0]) / projector_resolution[1];  // Aspect retio
    fov_horizontal_tof = fov_horizontal_tof_;  // TOF sensor's horizontal fov
    fov_vertical_tof = fov_vertical_tof_;  // TOF sensor's vertical fov
}

void TOFCorrection::correct() {
    calculate_xyz();  // Calculate points' 3D coordinates in TOF sensor's frame
    convert_xyz2projector();  // Convert points' xyz to projector's frame
    fit_plane_projector();  // Fit plane using points' 3D coordinates in projector's frame
    calculate_Kp(); // Calculate projector's intrinsic matrix using EOF_angle, throw_ratio and resolution
    calculate_corners_wall();  // Calculate projection image's corners in wall using plane coefficients and projector's params
    get_H_wall2p();  // Get homography matrix from wall to projector image pixel
    find_inscribed_rect();  // Find inscribed rectangle in wall's display region
    calculate_corrected_corners();  // Calculate corrected corners in projector's image pixel
    calculate_offsets();  // Calculate projection image's corners offsets in pixel
}

void TOFCorrection::update(const cv::Mat &tof_data_, const cv::Vec3d &gravity_using_) {
    // If you want to update the correction
    tof_data = tof_data_;
    gravity_using = gravity_using_;
    correct();
}

void TOFCorrection::calculate_xyz() {
    // Calculate points' 3D coordinates in TOF sensor's frame
    double fov_horizontal_rad = fov_horizontal_tof / 180 * PI;
    double fov_vertical_rad = fov_vertical_tof / 180 * PI;
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
    points_xyz_tof = cv::Mat::zeros(cv::Size(3, int(points_xyz_temp.size())), CV_64FC1);
    for (int i = 0; i < points_xyz_temp.size(); i++) {
        for (int j = 0; j < 3; j++) {
            points_xyz_tof.at<double>(i, j) = points_xyz_temp[i][j];
        }
    }
}


void TOFCorrection::convert_xyz2projector() {
    // Convert points' xyz to world's frame
    // Convert using rotation vector
    cv::Mat R_w2wall;
    cv::Rodrigues(rotation_vector_tof2projector, R_w2wall);
    points_xyz_projector = (R_w2wall * points_xyz_tof.t()).t();  // N*3
}

void TOFCorrection::fit_plane_projector() {
    // Use points_xyz_projector to fit a plane
    // points_xyz_projector is a CV_64F cv::Mat, N*3
    // ax + by + c = z
    // A = [x, y, 1]
    // X = [a, b, c].T
    // B = [z]
    // Solve Ax = B
    // Construct A
    int N = points_xyz_projector.rows;
    cv::Mat A;
    cv::Rect roi_xy(0, 0, 2, N);
    cv::Rect roi_z(2, 0, 1, N);
    cv::Mat ones = cv::Mat::ones(N, 1, CV_64F);
    cv::hconcat(points_xyz_projector(roi_xy), ones, A);
    // Construct B
    cv::Mat B = points_xyz_projector(roi_z);
    // Fit
    cv::solve(A, B, plane_coefficients, cv::DECOMP_SVD);
}

void TOFCorrection::calculate_Kp() {
    // Calculate projector's intrinsic matrix using EOF_angle, throw_ratio and resolution
    double fx, fy, cx, cy;
    cx = resolution_p[0] / 2.0;
    fx = throw_ratio / 0.5 * cx;
    fy = fx;
    cy = resolution_p[1] + tan(EOF_angle / 180.0 * PI) * fy;
    Kp = (cv::Mat_<double>(3, 3)
            <<
            fx, 0, cx,
            0, fy, cy,
            0, 0, 1);
}

void TOFCorrection::calculate_corners_wall() {
    // Calculate projection image's corners in wall using plane coefficients and projector's params
    // ax + by + c = z
    // plane_coefficients = {a, b, c}
    // We want plane: Ax+By+Cz+D=0
    std::vector<double> plane = {plane_coefficients[0], plane_coefficients[1], -1, 1};
    cv::Mat Kp_I;
    cv::invert(Kp, Kp_I);
    cv::Mat corners_;
    corners_ = Kp_I * (cv::Mat_<double>(3, 4)
            <<
            0, resolution_p[0], resolution_p[0], 0,
            0, 0, resolution_p[1], resolution_p[1],
            1, 1, 1, 1);

    cv::Mat corners_projector_xyz = cv::Mat::zeros(4, 3, CV_64F);
    for (int i = 0; i <= 3; i++) {
        cv::Vec3d corner(corners_.at<double>(0, i), corners_.at<double>(1, i), corners_.at<double>(2, i));
        cv::Vec3d corner_projector_xyz = findIntersection(corner, plane);
        for (int j = 0; j <= 2; j++) {
            corners_projector_xyz.at<double>(i, j) = corner_projector_xyz[j];
        }
    }

    // Convert corners to world
    // Use gravity vector cross product to calculate rotation vector
    cv::Vec3d G_cross = gravity_using.cross(gravity_standard);
    double theta = std::asin(cv::norm(G_cross) / (cv::norm(gravity_standard) * cv::norm(gravity_using)));
    cv::Vec3d rotation_vector_p2w = G_cross * (theta / cv::norm(G_cross));
    // Convert rotation vector to rotation matrix
    cv::Mat R_p2w;

    bool hasNaN = std::isnan(rotation_vector_p2w[0]) ||
                  std::isnan(rotation_vector_p2w[1]) ||
                  std::isnan(rotation_vector_p2w[2]);

    if (hasNaN)
    {
        R_p2w = cv::Mat::eye(3,3,CV_64F);
    }
    else{
        cv::Rodrigues(rotation_vector_p2w, R_p2w);
    }

    // Use rotation matrix to execute coordinate conversion
    cv::Mat corners_xyz_w = (R_p2w * corners_projector_xyz.t()).t();  // N*3
    // Convert corners from world frame to wall frame

    // Normal vector of plane
    cv::Vec3d n_plane_projector(plane[0], plane[1], -1);
    cv::Mat n_plane_world_mat = (R_p2w * n_plane_projector).t();
    cv::Vec3d n_plane_world;
    for (int i=0; i<3;i++)
    {
        n_plane_world[i] = n_plane_world_mat.at<double>(0,i);
    }


    double n_norm = cv::norm(n_plane_world); // Magnitude of normal vector
    // Wall's normal vector in wall frame
    cv::Vec3d n_wall(0.0, 0.0, -n_norm);
    // Calculate rotation vector by vector's cross product
    cv::Vec3d N_cross = n_plane_world.cross(n_wall);
    theta = std::asin(cv::norm(N_cross) / std::pow(n_norm, 2));
    cv::Vec3d rotation_vector_w2wall = N_cross * (theta / cv::norm(N_cross));


    // Convert rotation vector to rotation matrix
    cv::Mat R_w2wall;
    hasNaN = std::isnan(rotation_vector_w2wall[0]) ||
                  std::isnan(rotation_vector_w2wall[1]) ||
                  std::isnan(rotation_vector_w2wall[2]);

    if (hasNaN)
    {
        R_w2wall = cv::Mat::eye(3,3,CV_64F);
    }
    else{
        cv::Rodrigues(rotation_vector_w2wall, R_w2wall);
    }

    // Projection kps 3D coordinates to 2D wall frame
    cv::Mat corners_xyz_wall = (R_w2wall * corners_xyz_w.t()).t();
    cv::Rect roi_xy(0, 0, 2, 4);
    corners_wall = corners_xyz_wall(roi_xy);
    std::cout << "Projection points" << std::endl;
    std::cout << corners_wall << std::endl;
}


void TOFCorrection::get_H_wall2p() {
    // Find homography from wall to projector image
    cv::Mat corners_p = (cv::Mat_<double>(4, 2)
            <<
            0, 0,
            resolution_p[0], 0,
            resolution_p[0], resolution_p[1],
            0, resolution_p[1]);

    H_wall2p = cv::findHomography(corners_wall, corners_p);
}

void TOFCorrection::find_inscribed_rect() {
    // Get inscribed rectangle in certain aspect ratio of display area in wall frame
    // A very elegant approach, but may be buggy
    // Do not focus on details
    std::vector<cv::Point2d> corners;
    std::vector<cv::Point2d> inscribed_corners(4);

    for (int i = 0; i < corners_wall.rows; ++i) {
        double x = corners_wall.at<double>(i, 0);
        double y = corners_wall.at<double>(i, 1);
        corners.emplace_back(x, -y);
    }
    bool flip_horizontal = false;
    bool flip_vertical = false;
    if (((distance(corners[1], corners[2]) > distance(corners[0], corners[3])) and biggest)
        or ((distance(corners[1], corners[2]) < distance(corners[0], corners[3])) and !biggest)) {
        flip_horizontal = true;
        for (int i = 0; i <= 3; i++) { corners[i].x = -corners[i].x; }
        std::swap(corners[0], corners[1]);
        std::swap(corners[2], corners[3]);
    }

    int count = 0;
    while (count < 2) {
        cv::Point2d A = corners[0];
        cv::Point2d B = corners[1];
        cv::Point2d C = corners[2];
        cv::Point2d D = corners[3];
        std::vector<double> k = {slope(A, B), slope(D, A), slope(D, C)};
        if (k[0] > 0 and k[1] > 0 and k[2] > 0) {
            inscribed_corners[0] = A;
            inscribed_corners[2] = findPointInLineByAspectRatio(D, C, A, aspect_ratio);
            inscribed_corners[1] = cv::Point2d(inscribed_corners[2].x, inscribed_corners[0].y);
            inscribed_corners[3] = cv::Point2d(inscribed_corners[0].x, inscribed_corners[2].y);
            break;
        } else if (k[0] > 0 and k[1] > 0 and k[2] < 0) {
            inscribed_corners[0] = A;
            inscribed_corners[3] = findPointInLineByX(D, C, A.x);
            double h = std::abs(inscribed_corners[0].y - inscribed_corners[3].y);
            inscribed_corners[1] = cv::Point2d(A.x + aspect_ratio * h, A.y);
            inscribed_corners[2] = cv::Point2d(inscribed_corners[1].x, inscribed_corners[3].y);
            break;
        } else if (k[0] > 0 and k[1] < 0 and k[2] > 0) {
            double x = ((aspect_ratio * (k[1] - k[0]) * A.x) - ((1 / k[2] - 1 / k[1]) * (A.y - D.y - k[1] * A.x))) / (
                    (1 / k[2] - 1 / k[1]) * k[1] - aspect_ratio * (k[0] - k[1]));
            double y = findPointInLineByX(A, D, x).y;
            double w = (1 / k[2] - 1 / k[1]) * (y - D.y);
            double h = (k[0] - k[1]) * (x - A.x);
            inscribed_corners[0] = cv::Point2d(x, y + h);
            inscribed_corners[1] = cv::Point2d(x + w, y + h);
            inscribed_corners[2] = cv::Point2d(x + w, y);
            inscribed_corners[3] = cv::Point2d(x, y);
            break;
        } else if (k[0] < 0 and k[1] > 0 and k[2] > 0) {
            double xi = findIntersection(A, k[0], D, k[2]).x;
            double x = ((1 / k[1] - 1 / k[0]) * k[0] * A.x + aspect_ratio * (k[2] - k[0]) * xi) / (
                    (aspect_ratio * (k[2] - k[0])) + (1 / k[1] - 1 / k[0]) * k[0]);
            double y = findPointInLineByX(A, B, x).y;
            double w = x - findPointInLineByY(A, D, y).x;
            double h = y - findPointInLineByX(D, C, x).y;
            inscribed_corners[0] = cv::Point2d(x - w, y);
            inscribed_corners[1] = cv::Point2d(x, y);
            inscribed_corners[2] = cv::Point2d(x, y - h);
            inscribed_corners[3] = cv::Point2d(x - w, y - h);
            break;
        }
        flip_vertical = true;
        for (int i = 0; i <= 3; i++) { corners[i].y = -corners[i].y; }
        std::swap(corners[0], corners[3]);
        std::swap(corners[1], corners[2]);
        count++;
        // TODO: Make the algorithm more robust
        if (count == 2) {
            throw std::runtime_error("FindInscribeRectangleError");
        }
    }
    if (flip_vertical) {
        for (int i = 0; i <= 3; i++) { corners[i].y = -corners[i].y; }
        std::swap(corners[0], corners[3]);
        std::swap(corners[1], corners[2]);
        for (int i = 0; i <= 3; i++) { inscribed_corners[i].y = -inscribed_corners[i].y; }
        std::swap(inscribed_corners[0], inscribed_corners[3]);
        std::swap(inscribed_corners[1], inscribed_corners[2]);
    }
    if (flip_horizontal) {
        for (int i = 0; i <= 3; i++) { corners[i].x = -corners[i].x; }
        std::swap(corners[0], corners[1]);
        std::swap(corners[2], corners[3]);
        for (int i = 0; i <= 3; i++) { inscribed_corners[i].x = -inscribed_corners[i].x; }
        std::swap(inscribed_corners[0], inscribed_corners[1]);
        std::swap(inscribed_corners[2], inscribed_corners[3]);
    }
    for (int i = 0; i <= 3; i++) { corners[i].y = -corners[i].y; }
    for (int i = 0; i <= 3; i++) { inscribed_corners[i].y = -inscribed_corners[i].y; }

    inscribed_rect_corners_wall = (cv::Mat_<double>(4, 2)
            <<
            inscribed_corners[0].x, inscribed_corners[0].y,
            inscribed_corners[1].x, inscribed_corners[1].y,
            inscribed_corners[2].x, inscribed_corners[2].y,
            inscribed_corners[3].x, inscribed_corners[3].y);
    std::cout << "Inscribed rectangle corners:" << std::endl;
    std::cout << inscribed_rect_corners_wall << std::endl;
}

void TOFCorrection::calculate_corrected_corners() {
    // From wall's inscribed rectangle corners to projector's image, using homography matrix
    bool in_range = false;
    int count = 0;
    while (!in_range) {
        cv::Mat inscribed_corners_wall_;
        cv::hconcat(inscribed_rect_corners_wall, cv::Mat::ones(4, 1, CV_64F), inscribed_corners_wall_);
        cv::Mat corrected_corners_p_T = H_wall2p * inscribed_corners_wall_.t();
        corrected_corners_p = normalize(corrected_corners_p_T);
        in_range = true;
        for (int i = 0; i <= 3; i++) {
            if (!((0 < corrected_corners_p.at<double>(i, 0) and corrected_corners_p.at<double>(i, 0)< resolution_p[0]) and
                  (0 < corrected_corners_p.at<double>(i, 1) and corrected_corners_p.at<double>(i, 1)< resolution_p[1]))) {
                in_range = false;
                break;
            }
        }
        if (!in_range) {
            inscribed_rect_corners_wall = shrink(inscribed_rect_corners_wall);
        }
        count++;
        if (count == 100) {
            throw std::runtime_error("ShrinkError");
        }
    }
    std::cout << "Corrected display region:" << std::endl;
    std::cout << corrected_corners_p << std::endl;
}


void TOFCorrection::calculate_offsets() {
    // Calculate four corners' pixel offsets
    offsets = corrected_corners_p - (cv::Mat_<double>(4, 2)
            <<
            0, 0,
            resolution_p[0], 0,
            resolution_p[0], resolution_p[1],
            0, resolution_p[1]);
}

