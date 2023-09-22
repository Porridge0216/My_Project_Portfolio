#include <ros/ros.h> 
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <netinet/in.h>//for sockaddr_in
#include <arpa/inet.h>//for socket 
#include <communication/packet_serial.hpp>
// #include <communication/newpacket.hpp>
// #include <find_cylinder/CylinderParam.h>
#include <pthread.h>
#include <std_msgs/UInt8.h>
#include <std_msgs/Int32.h>
#include <geometry_msgs/Pose.h>
#include <sensor_msgs/Image.h>
#define reply 0x02
#define pcdata 0x01
#define needreply 0x00
#define step 2997*3 // 发送图像拆包后的大小
using namespace std;
void UDPsend(uint8_t* ,uint16_t );
void image_UDPsend(uint8_t* ,uint16_t );
bool UDPinit();
void* UDPreviceve(void* args);
static int fd;
static struct sockaddr_in addr_to;//目标服务器地址
static struct sockaddr_in image_addr_to;
static struct sockaddr_in addr_from;

//br_packet::Packet packet;
br_packet::Packet packet;
bool UDPhand();
ros::Publisher setloc_pub;
ros::Publisher reset_pub;
ros::Publisher findball_pub;
ros::Publisher findblock_pub;
ros::Publisher square_mod_pub;
ros::Subscriber ballpos_sub;
ros::Subscriber blockpos_sub;
ros::Subscriber image_sub;
std::string findblock_topic,findball_topic;
std::string blockpos_topic, ballpos_topic;
std::string image_topic;
void findblock_callback(uint8_t* ,uint16_t );
void findball_callback(uint8_t* ,uint16_t );
void square_mod_callback(uint8_t* ,uint16_t );
// void blockpos_callback(const find_cylinder::CylinderParam& msg);

int circle_R;
std::string to_ip,from_ip;
int to_hton, from_hton, image_to_hton;
bool image_need = false;
// std::vector<find_cylinder::CylinderParam> cylinder_msg;
int square_mod=0;
double cx,cy,fx,fy,depth;
//int image_int=0;

