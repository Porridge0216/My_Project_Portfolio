//
// Created by Porridge on 2023/9/8.
//

#ifndef KEYSTONECORRECTION_H
#define KEYSTONECORRECTION_H

#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <iostream>
#include <vector>


class KeystoneCorrection {
    /* Auto Keystone-Correction Projector Based on Structured Light Pair
     * Once you instantiate the class:
     * You can get the four cornerstones (in projector image frame) and projector rotation (in world's frame) directly
     * If you need a new correction, please use update(), offsets and rotation will be updated automatically*/
public:
    KeystoneCorrection(const cv::Mat &camera_intrinsic_mtx, const cv::Mat &camera_distortion,
                       const cv::Mat &projector_intrinsic_mtx, const cv::Mat &projector_distortion,
                       const std::vector<int> &projector_resolution, const cv::Mat &r_cinp, const cv::Mat &t_cinp,
                       const cv::Vec3d &gravity_vector_standard, const cv::Vec3d &gravity_vector_using,
                       const cv::Mat &img_camera, const cv::Mat &img_projector, bool biggest_or_clearest);

    void update(const cv::Mat &img_camera,
                const cv::Vec3d &gravity_vector_using);  // Update correction, offsets and rotation will be updated


    cv::Mat offsets;  // Four corners' pixel offsets in OpenCV coordinate system 4*2
    cv::Mat rotation_matrix;  // Rotation matrix describes projector in world's frame 3*3


private:

    cv::Mat Kc; // Camera intrinsic matrix 3*3
    cv::Mat Kp;  // Projector intrinsic matrix 3*3
    cv::Mat Dp;  // Projector distortion coefficients 1*5
    cv::Mat Dc;  // Camera distortion coefficients 1*5
    std::vector<int> resolution_p;  // Projector resolution width_pix, height_pix
    double aspect_ratio;  // Projector aspect ratio (width/height)
    cv::Mat R_cinp;  // Rotation matrix describing the camera in the projector coordinate system 3*3
    cv::Mat T_cinp;  // Translation vector describing the camera in the projector coordinate system 3*1
    cv::Vec3d gravity_vector_p_standard;  // Accelerometer values in standard horizontal state, in projector frame
    cv::Vec3d gravity_vector_p;  // Accelerometer values in the current state, in projector frame
    cv::Mat img_c;  // Undistorted camera image
    cv::Mat img_p;  // Undistorted projector image
    bool biggest;  // Biggest or clearest

    cv::Mat kps_p;  // Key points in projector image 68*2
    cv::Mat kps_c;  // Key points in camera image 68*2
    cv::Mat kps_xyz_p; // 3D kps in projector frame 68*3
    cv::Mat kps_xyz_w; // 3D kps in world frame 68*3
    cv::Mat kps_xy_wall;  // 2D kps in wall frame 68*2
    cv::Vec3d plane_params; // plane coefficients a,b,c in ax + by + c = z,
    cv::Mat H_wall2p;  // Homography from wall to projector image
    cv::Mat corners_wall;  // Predicted display area's corners in wall frame
    cv::Mat inscribed_rect_corners_wall;  // Corners of the inscribed rectangle in wall frame
    cv::Mat corrected_corners_p;  // Corrected corners of display area in projector image pixel frame

    static cv::Mat getKeyPoints(const cv::Mat &img);  // Get key-points(chessboard corners, apriltags) in image
    void getKpsXYZInProj();  // Get key-points'3D coordinates in projector by structured light pair
    void getKpsXYZInWorld();  // Get key-points'3D coordinates in world using gravity difference
    void getPlane();  // Get plane using all the 3D key-points in world
    void getKpsXYWall();  // Get key-points'2D coordinates in wall
    inline void getH();  // Get homograph from wall to projector's image
    void getCornersWall();  // Predict the display area's corners in wall frame
    void getInscribedRect();  // Get the inscribed rectangle of display area in wall
    void getCorrectedCornersP();  // Get corrected corners in projector's image
    inline void getOffsets();  // Get four corners' pixel offsets in OpenCV coordinate system

    void correct();  // Execute correction using all the functions above
    void getPose();  // Get projector's pose in world frame
};

#endif //KEYSTONECORRECTION_H
