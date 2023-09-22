//
// Created by Porridge on 2023/9/16.
//

#ifndef TOF_TOFCORRECTION_H
#define TOF_TOFCORRECTION_H

#include <opencv2/core.hpp>
#include <opencv2/calib3d.hpp>

class TOFCorrection {
public:
    TOFCorrection(const cv::Mat &tof_data_, const cv::Vec3d &rotation_vector_tof2projector_,
                  const cv::Vec3d &gravity_standard_, const cv::Vec3d &gravity_using_,
                  const double &EOF_angle_, const double &throw_ratio_, bool biggest_,
                  const std::vector<int> &projector_resolution = {1920, 1080},
                  const double &fov_horizontal_tof_ = 45.0, const double &fov_vertical_tof_ = 45.0);

    void correct();

    void update(const cv::Mat &tof_data_, const cv::Vec3d &gravity_using_);

    cv::Mat offsets;

private:
    cv::Mat tof_data;
    cv::Vec3d rotation_vector_tof2projector;
    cv::Vec3d gravity_standard;
    cv::Vec3d gravity_using;
    double EOF_angle;
    double throw_ratio;
    bool biggest;

    std::vector<int> resolution_p;

    double fov_horizontal_tof;
    double fov_vertical_tof;
    double aspect_ratio;

    cv::Mat points_xyz_tof;
    cv::Mat points_xyz_projector;
    cv::Vec3d plane_coefficients;

    cv::Mat corners_wall;
    cv::Mat inscribed_rect_corners_wall;
    cv::Mat H_wall2p;
    cv::Mat Kp;
    cv::Mat corrected_corners_p;

    void calculate_xyz();

    void convert_xyz2projector();

    void fit_plane_projector();

    void calculate_Kp();

    void calculate_corners_wall();

    void get_H_wall2p();

    void find_inscribed_rect();

    void calculate_corrected_corners();

    void calculate_offsets();
};


#endif //TOF_TOFCORRECTION_H
