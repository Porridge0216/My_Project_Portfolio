#include <communication/communicator.hpp>



int main(int argc, char** argv)
{
    ros::init (argc,argv,"communicator_test");
    ros::NodeHandle nh;
    ros::NodeHandle nh_local("~");
    Communicator communicator_(nh, nh_local);
    
    ros::Rate loop_rate(60);
    

    while(ros::ok())
    {
        communicator_.robotPacket.update();
        communicator_.controllerPacket.update();
        ros::spinOnce();
        loop_rate.sleep();
    }
}
