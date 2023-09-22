#ifndef Com
#define Com

#include <ros/ros.h> 
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <netinet/in.h>//for sockaddr_in
#include <arpa/inet.h>//for socket 
#include <communication/packet.hpp>
#include <thread>
#include <serial/serial.h>
#include <tf/transform_broadcaster.h>
#include <std_msgs/UInt8.h>
#include <std_msgs/Int32.h>
#include <pthread.h>
#include <communication/ring.h>
#include <communication/rings.h>
#include <communication/shoot_aid.h>
#include <communication/set_field.h>
#define reply 0x02
#define pcdata 0x01
#define needreply 0x00

namespace br_packet {
    class Packet;
}
// class br_packet::Packet;
class Communicator
{
    public:
        // Communicator();
        Communicator(ros::NodeHandle&,ros::NodeHandle&);
        static void UDPsend(uint8_t* ,uint16_t ,target*);
        void loadParameters();
        void UDPinit();
        static void robotUDPrece(target_ *,br_packet::Packet*);
        static void controllerUDPrece(target_ *,br_packet::Packet*);
        void rosInit();
        void exampleCallback(uint8_t*, uint16_t);
        void example_ros_Callback(const std_msgs::UInt8&);
        static void example_robot_port1_Callback(uint8_t*, uint16_t);
        static void set_field_Callback(uint8_t*, uint16_t);
        static void example_controller_port0_Callback(uint8_t*, uint16_t);
        void port2_sub_ros_Callback(const communication::rings&);
        static void controller_port2_Callback(uint8_t*, uint16_t);
        static void controller_port3_Callback(uint8_t*,uint16_t);
        target controller,robot;
        br_packet::Packet robotPacket,controllerPacket; 
        ros::NodeHandle nh_,nh_local_;
    private:
        
        static ros::Publisher example_pub;
        static ros::Publisher port2_pub;
        static ros::Publisher port3_pub;
        static ros::Subscriber example_sub;
        static ros::Subscriber port2_sub;
        static ros::ServiceClient set_field_client;
        std::string example_pub_topic;
        std::string example_sub_topic;
        std::string port2_sub_topic;
        std::string port2_pub_topic;
        std::string port3_pub_topic;
        std::string set_field_srv;

};
// br_packet::Packet robotPacket,comtrollerPacket;
ros::Publisher Communicator::example_pub;
ros::Publisher Communicator::port2_pub;
ros::Publisher Communicator::port3_pub;
ros::Subscriber Communicator::example_sub;
ros::Subscriber Communicator::port2_sub;
ros::ServiceClient Communicator::set_field_client;
#endif