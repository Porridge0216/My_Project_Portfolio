#include "ekf_pose_fusion/ekf_pose_fusion.hpp"

using namespace estimation;
int main(int argc, char **argv)
{
    // Initialize ROS
    ros::init(argc, argv, "robot_pose_ekf");

    // create filter class
    BR_pose_ekf my_filter_node(2);

    // ros::MultiThreadedSpinner spinner(3);
    ros::AsyncSpinner spin(2);
    spin.start();

    // spinner.spin();
    ros::waitForShutdown();

    return 0;
}