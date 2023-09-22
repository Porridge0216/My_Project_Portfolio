#include <ros/ros.h> 
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <netinet/in.h>//for sockaddr_in
#include <arpa/inet.h>//for socket 
#include <serial/serial.h>
#include <std_msgs/String.h>
#include <pthread.h>
using namespace std;


bool UDP_init();
bool serialinit();
void UDPreviceve();
void UDPsend();
void serialread();
bool UDPhand();

int fd, r;
serial::Serial ser;
struct sockaddr_in addr_to;//目标服务器地址
struct sockaddr_in addr_from;

std::string to_ip,from_ip;
int to_hton, from_hton;
std_msgs::String exam_msg;