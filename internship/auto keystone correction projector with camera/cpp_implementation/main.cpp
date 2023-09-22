#include <opencv2/opencv.hpp>
#include "KeystoneCorrection.h"

int main() {
    // Images of camera and projector
    cv::Mat image_camera = cv::imread("../uc7.jpg");
    cv::Mat image_projector = cv::imread("../up.jpg");

    // Camera intrinsic matrix and distortion coefficients
    cv::Mat camera_intrinsic_mtx = (cv::Mat_<double>(3, 3)
            <<
            600.32380483, 0, 323.20302011,
            0., 600.39865037, 202.07122001,
            0., 0., 1);

    cv::Mat camera_dist_coefs = (cv::Mat_<double>(1, 5)
            << 2.48974585e-01, -1.49624210e+00, -4.27611936e-04, 5.42105284e-03, 1.84425912e+00);

    // Projector intrinsic matrix and distortion coefficients
    cv::Mat projector_intrinsic_mtx = (cv::Mat_<double>(3, 3)
            <<
            2.28382693e+03, 0.00000000e+00, 9.89334268e+02,
            0.00000000e+00, 2.28805040e+03, 1.07939241e+03,
            0.00000000e+00, 0.00000000e+00, 1.00000000e+00);

    cv::Mat projector_dist_coefs = (cv::Mat_<double>(1, 5)
            << 0.03103913, 0.02877732, -0.00038886, 0.00115072, -0.31241419);

    // Projector resolution
    std::vector<int> projector_resolution = {1920, 1080};

    // Rotation matrix describing camera in projector
    cv::Mat R_cinp = (cv::Mat_<double>(3, 3)
            <<
            0.99970737, -0.00727533, 0.02307026,
            0.01073077, 0.98810691, -0.15339354,
            -0.0216799, 0.15359621, 0.98789584);

    // Translation vector describing camera in projector
    cv::Mat t_cinp = (cv::Mat_<double>(3, 1)
            <<
            -68.82968479,
            0.40231569,
            7.98265215);

    // Standard gravity vector in projector's coordinate frame
    cv::Mat gravity_standard = (cv::Mat_<double>(3, 1)
            <<
            2.5,
            7,
            0.45);

    // Gravity vector in projector's coordinate frame now
    cv::Mat gravity_using = (cv::Mat_<double>(3, 1)
            <<
            0.01,
            9.8,
            0.01);

    // Instantiate the KeystoneCorrection class by all values above
    KeystoneCorrection correction(camera_intrinsic_mtx, camera_dist_coefs, projector_intrinsic_mtx,
                                  projector_dist_coefs, projector_resolution, R_cinp, t_cinp, gravity_standard,
                                  gravity_using, image_camera, image_projector, false);

    // Require offsets and pose from the corrector
    cv::Mat offsets1 = correction.offsets;  // Four corners' pixel offsets, follow the opencv coordinate system
    cv::Mat rotation1 = correction.rotation_matrix;  // rotation_matrix of projector, in world's coordinate frame

    // You can execute a new correction in this way
    correction.update(image_camera, gravity_using);

    cv::Mat offsets2 = correction.offsets;
    cv::Mat rotation2 = correction.rotation_matrix;

}
