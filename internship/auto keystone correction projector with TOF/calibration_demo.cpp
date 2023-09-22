#include <iostream>
#include "calibrationTOF.h"
int main() {
    cv::Mat tof_data = (cv::Mat_<int>(8, 8) << 1127, 1062, 1036, 977, 976, 993, 1013, 1092,
            776, 746, 706, 687, 716, 694, 719, 749,
            588, 564, 544, 539, 540, 547, 553, 566,
            490, 480, 478, 479, 471, 475, 470, 477,
            433, 426, 425, 423, 423, 418, 420, 420,
            381, 380, 378, 378, 377, 373, 370, 370,
            338, 341, 341, 339, 338, 334, 334, 328,
            304, 306, 304, 305, 303, 302, 298, 297
    );
    calibrationTOF calibration(tof_data);
    cv::Vec3d rotation_vector_tof2projector;
    calibration.calibrate(rotation_vector_tof2projector);
    std::cout << rotation_vector_tof2projector << std::endl;
}
