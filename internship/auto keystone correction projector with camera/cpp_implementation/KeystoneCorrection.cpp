//
// Created by Porridge on 2023/9/8.
//

#include "KeystoneCorrection.h"
#include "generalFunctions.h"
#include "apriltag/tag16h5.h"
#include <cmath>
#include <iostream>


KeystoneCorrection::KeystoneCorrection(const cv::Mat &camera_intrinsic_mtx, const cv::Mat &camera_distortion,
                                       const cv::Mat &projector_intrinsic_mtx, const cv::Mat &projector_distortion,
                                       const std::vector<int> &projector_resolution,
                                       const cv::Mat &r_cinp, const cv::Mat &t_cinp,
                                       const cv::Vec3d &gravity_vector_standard,
                                       const cv::Vec3d &gravity_vector_using, const cv::Mat &img_camera,
                                       const cv::Mat &img_projector,
                                       bool biggest_or_clearest) {
    Kc = camera_intrinsic_mtx;
    Kp = projector_intrinsic_mtx;
    Dp = projector_distortion;
    Dc = camera_distortion;
    resolution_p = projector_resolution;
    aspect_ratio = static_cast<double>(projector_resolution[0]) / projector_resolution[1];
    R_cinp = r_cinp;
    T_cinp = t_cinp;
    gravity_vector_p_standard = gravity_vector_standard;
    gravity_vector_p = gravity_vector_using;
    cv::undistort(img_camera, img_c, camera_intrinsic_mtx, camera_distortion);
    cv::undistort(img_projector, img_p, projector_intrinsic_mtx, projector_distortion);
    biggest = biggest_or_clearest;

    correct();
    getPose();
}


cv::Mat KeystoneCorrection::getKeyPoints(const cv::Mat &img) {
    cv::Mat kps;
    // Find chessboard
    cv::Size pattern_size(8, 6);
    cv::Mat chessboard_corners_float;  // CV_32FC2
    bool chessboard_detected = cv::findChessboardCorners(img, pattern_size, chessboard_corners_float,
                                                         cv::CALIB_CB_ADAPTIVE_THRESH);


    cv::Mat img_gray;
    cv::cvtColor(img,img_gray,cv::COLOR_BGR2GRAY);

    cv::find4QuadCornerSubpix(img_gray, chessboard_corners_float, cv::Size(13, 13));

    // Judge if found chessboard
    if (!chessboard_detected) { throw std::runtime_error("ChessboardNotFoundError"); }

    cv::drawChessboardCorners(img,pattern_size,chessboard_corners_float, true);

    cv::imshow("res",img);
    cv::waitKey(0);



    // Convert detection result to double
    cv::Mat chessboard_corners;
    chessboard_corners_float.convertTo(chessboard_corners, CV_64FC2);

    // Detect apriltags
    // Input image needs to be gray

    // Convert opencv img to apriltag img
    image_u8_t im = {.width = img_gray.cols,
            .height = img_gray.rows,
            .stride = img_gray.cols,
            .buf = img_gray.data
    };
    // Detect
    apriltag_detector_t *td = apriltag_detector_create();
    apriltag_family_t *tf = tag16h5_create();
    apriltag_detector_add_family(td, tf);
    zarray_t *detections = apriltag_detector_detect(td, &im);
    cv::Mat apriltags(20, 1, CV_64FC2);  // cv::Mat to save apriltags points
    // Save detect results to cv::Mat
    // TODO: Make the algorithm more robust
    int apriltag_useful_count = 0;
    for (int i = 0; i < zarray_size(detections); i++) {
        apriltag_detection_t *det;
        zarray_get(detections, i, &det);
        int id_shift = det->id - 26;
        if (id_shift < 0 or id_shift > 3) {
            continue;
        }
        // Copy corners to the Mat
        apriltag_useful_count++;
        for (int j = 0; j < 4; j++) {
            apriltags.at<cv::Vec2d>(id_shift * 5 + j, 0)[0] = det->p[j][0];
            apriltags.at<cv::Vec2d>(id_shift * 5 + j, 0)[1] = det->p[j][1];
        }
        // Copy center to the Mat
        apriltags.at<cv::Vec2d>(id_shift * 5 + 4, 0)[0] = det->c[0];
        apriltags.at<cv::Vec2d>(id_shift * 5 + 4, 0)[1] = det->c[1];
    }

    if (apriltag_useful_count != 4) {
        throw std::runtime_error("ApriltagsNotFoundError");
    }
    tag16h5_destroy(tf);
    apriltag_detector_destroy(td);
    // Concat chessboard corners and apriltag points to kps
    cv::vconcat(chessboard_corners, apriltags, kps);
    return kps; // CV_64FC2
}

void KeystoneCorrection::getKpsXYZInProj() {
    // Get 3D kps coordinate in projector, using triangulate
    // Calculate projection matrices
    cv::Mat Rt, Rt01;
    cv::hconcat(R_cinp, T_cinp, Rt);
    cv::Mat zero_one = (cv::Mat_<double>(1, 4) << 0, 0, 0, 1);
    cv::vconcat(Rt, zero_one, Rt01);
    cv::Mat Rt_I;  // Projector inscribed in camera
    cv::invert(Rt01, Rt_I);
    // Camera projection matrix 3*4
    cv::Rect roi(0, 0, 4, 3);
    cv::Mat projection_matrix_c = Kc * Rt_I(roi);
    // Projector projection matrix 3*4
    cv::Mat projection_matrix_p;
    cv::Mat zeros(3, 1, CV_64F, cv::Scalar(0.0));
    cv::hconcat(Kp, zeros, projection_matrix_p);
    // Triangulate
    cv::Mat kps_xyzwT; // Homogeneous coordinate 4*68
    cv::triangulatePoints(projection_matrix_p, projection_matrix_c, kps_p.t(), kps_c.t(), kps_xyzwT);
    // Normalize
    kps_xyz_p = normalize(kps_xyzwT);  // 68*3
};

void KeystoneCorrection::getKpsXYZInWorld() {
    // Convert kps from projector frame to world frame
    // Use gravity vector cross product to calculate rotation vector
    cv::Vec3d G_cross = gravity_vector_p.cross(gravity_vector_p_standard);
    double theta = std::asin(cv::norm(G_cross) / (cv::norm(gravity_vector_p_standard) * cv::norm(gravity_vector_p)));
    cv::Vec3d rotation_vector_p2w = G_cross * (theta / cv::norm(G_cross));
    // Convert rotation vector to rotation matrix
    cv::Mat R_p2w;
    cv::Rodrigues(rotation_vector_p2w, R_p2w);
    // Use rotation matrix to execute coordinate conversion
    kps_xyz_w = (R_p2w * kps_xyz_p.t()).t();  // 68*3
};

void KeystoneCorrection::getPlane() {
    // Use kps_xyz_w to fit a plane
    // kps_xyz_w is a CV_64F cv::Mat, 68*3
    // ax + by + c = z
    // A = [x, y, 1]
    // X = [a, b, c].T
    // B = [z]
    // Solve Ax = B
    // Construct A
    cv::Mat A;
    cv::Rect roi_xy(0, 0, 2, 68);
    cv::Rect roi_z(2, 0, 1, 68);
    cv::Mat ones = cv::Mat::ones(68, 1, CV_64F);
    cv::hconcat(kps_xyz_w(roi_xy), ones, A);
    // Construct B
    cv::Mat B = kps_xyz_w(roi_z);
    // Fit
    cv::solve(A, B, plane_params, cv::DECOMP_SVD);
    std::cout << "Plane:" << std::endl;
    std::cout << plane_params[0] << " x + "
              << plane_params[1] << " y + "
              << plane_params[2] << " = z" << std::endl;
};

void KeystoneCorrection::getKpsXYWall() {
    // Convert kps from world frame to wall frame
    // Normal vector of plane
    cv::Vec3d n_plane(plane_params[0], plane_params[1], -1);
    double n_norm = cv::norm(n_plane); // Magnitude of normal vector
    // Wall's normal vector in wall frame
    cv::Vec3d n_wall(0.0, 0.0, -n_norm);
    // Calculate rotation vector by vector's cross product
    cv::Vec3d N_cross = n_plane.cross(n_wall);
    double theta = std::asin(cv::norm(N_cross) / std::pow(n_norm, 2));
    cv::Vec3d rotation_vector_w2wall = N_cross * (theta / cv::norm(N_cross));
    // Convert rotation vector to rotation matrix
    cv::Mat R_w2wall;
    cv::Rodrigues(rotation_vector_w2wall, R_w2wall);
    // Projection kps 3D coordinates to 2D wall frame
    cv::Mat kps_xyz_wall = (R_w2wall * kps_xyz_w.t()).t();
    cv::Rect roi_xy(0, 0, 2, 68);
    kps_xy_wall = kps_xyz_wall(roi_xy);
    std::cout << "Projection points" << std::endl;
    std::cout << kps_xy_wall << std::endl;
};

inline void KeystoneCorrection::getH() {
    // Find homography from wall to projector image
    H_wall2p = cv::findHomography(kps_xy_wall, kps_p);
};

void KeystoneCorrection::getCornersWall() {
    // Predict display area's corners in wall coordinate using homography matrix
    cv::Mat corners_p_ = (cv::Mat_<double>(4, 3)
            <<
            0, 0, 1,
            resolution_p[0], 0, 1,
            resolution_p[0], resolution_p[1], 1,
            0, resolution_p[1], 1);

    cv::Mat H_p2wall;
    cv::invert(H_wall2p, H_p2wall);
    cv::Mat corners_wall_T = H_p2wall * corners_p_.t(); // 3*4
    // Normalize
    corners_wall = normalize(corners_wall_T);  // 4*2
    std::cout << "Corners_wall:" << std::endl;
    std::cout << corners_wall << std::endl;
};

void KeystoneCorrection::getInscribedRect() {
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
};

void KeystoneCorrection::getCorrectedCornersP() {
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
            if (!((0 < corrected_corners_p.at<double>(i, 0) < resolution_p[0]) and
                  (0 < corrected_corners_p.at<double>(i, 1) < resolution_p[1]))) {
                in_range = false;
                break;
            }
        }
        if (!in_range) {
            inscribed_rect_corners_wall = shrink(inscribed_rect_corners_wall);
        }
        count++;
        if (count == 5) {
            throw std::runtime_error("ShrinkError");
        }
    }
    std::cout << "Corrected display region:" << std::endl;
    std::cout << corrected_corners_p << std::endl;
};

inline void KeystoneCorrection::getOffsets() {
    // Calculate four corners' pixel offsets
    offsets = corrected_corners_p - (cv::Mat_<double>(4, 2)
            <<
            0, 0,
            resolution_p[0], 0,
            resolution_p[0], resolution_p[1],
            0, resolution_p[1]);
}

void KeystoneCorrection::getPose() {
    // Calculate projector's pose, mainly rotation matrix
    cv::Mat obj_points;
    cv::hconcat(kps_xy_wall, cv::Mat::ones(68, 1, CV_64F), obj_points);
    cv::Vec3d r_vec, t_vec;
    cv::solvePnP(obj_points, kps_p, Kp, Dp, r_vec, t_vec);
    cv::Mat rotation_matrix_wall_in_p;
    cv::Rodrigues(r_vec, rotation_matrix_wall_in_p);
    cv::invert(rotation_matrix_wall_in_p, rotation_matrix);
}

void KeystoneCorrection::correct() {
    // Execute correction
    kps_p = getKeyPoints(img_p);
    kps_c = getKeyPoints(img_c);
    getKpsXYZInProj();
    getKpsXYZInWorld();
    getPlane();
    getKpsXYWall();
    getH();
    getCornersWall();
    getInscribedRect();
    getCorrectedCornersP();
    getOffsets();
}

void KeystoneCorrection::update(const cv::Mat &img_camera, const cv::Vec3d &gravity_vector_using) {
    // Update correction
    undistort(img_camera, img_c, Kc, Dc);
    gravity_vector_p = gravity_vector_using;
    correct();
    getPose();
}


