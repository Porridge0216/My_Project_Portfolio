//
// Created by Porridge on 2023/9/16.
//

#ifndef TOF_CALIBRATIONTOF_H
#define TOF_CALIBRATIONTOF_H

#include <opencv2/core.hpp>

class calibrationTOF {
public:
    explicit calibrationTOF(const cv::Mat &tof_data_,
                            const double &fov_horizontal_ = 45.0, const double &fov_vertical_ = 45.0);

    void calibrate(cv::Vec3d &rotation_vector_tof2projector_);

private:
    cv::Mat tof_data;
    double fov_horizontal;
    double fov_vertical;
    cv::Mat points_xyz;
    cv::Vec3d plane_coefficients_tof;
    cv::Vec3d rotation_vector_tof2projector;

    void calculate_xyz();

    void fit_plane();

    void calculate_rotation_vector();
};

#endif //TOF_CALIBRATIONTOF_H
