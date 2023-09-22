#include <ros/ros.h> 
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <netinet/in.h>//for sockaddr_in
#include <arpa/inet.h>//for socket 
// #include <communication/newpacket.hpp>
#include <serial/serial.h>
#include <pthread.h>
#include <communication/packet_serial.hpp>
#include <tf/transform_broadcaster.h>
// #include <find_cylinder/CylinderParam.h>
#include <communication/head_angle.h>
#include <communication/shoot_aid.h>
#include <std_msgs/UInt8.h>
#include <std_msgs/Int32.h>
#include <std_msgs/String.h>
#include <nav_msgs/Odometry.h>
#define reply 0x00
#define pcdata 0x01
#define needreply 0x02
using namespace std;
bool UDP_init();
bool serialinit();
static void write(uint8_t*, uint16_t);
void UDPreviceve();
void UDPsend();
bool UDPhand();
void packfloat(float f, uint8_t* da_ad);
void sendlocation();
void TFuodate(void*);
// void blockpos_callback(const find_cylinder::CylinderParam& msg);
void location_callback(uint8_t* ,uint16_t );
void img_angle_callback(uint8_t*,uint16_t);
void pos_callback(const nav_msgs::Odometry& msg);
void port0_callback(uint8_t*,uint16_t);
void shoot_aid_sub_callback(const std_msgs::UInt8& msg);
void* TFpub(void*);
serial::Serial ser;
br_packet::Packet packet;
uint8_t buff[1024],packbuff[100];
float x, y, u16yaw;
float xx, yy;
float yaw;
int len;
std::string base_name,world_name;
std::string img_topic;
std::string img_angle_topic;
std::string pos_topic;
std::string shoot_aid_topic;
std::string shoot_aid_srv;
// tf::StampedTransform base_world;
// tf::TransformBroadcaster tf_pub;
// tf::Quaternion q;
ros::Subscriber img_sub;
ros::Publisher img_pub;
ros::Subscriber pos_sub;
ros::Subscriber shoot_aid_sub;
ros::ServiceClient shoot_aid_client;
int fd, r;
struct sockaddr_in addr_to;//目标服务器地址
struct sockaddr_in addr_from;

std::string to_ip,from_ip;
int to_hton, from_hton;
std_msgs::String exam_msg;