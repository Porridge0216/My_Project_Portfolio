#include <ros/ros.h> 
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <netinet/in.h>//for sockaddr_in
#include <arpa/inet.h>//for socket 

typedef struct target{
    std::string Name_;
    std::string to_ip_,from_ip_;
    int to_hton_, from_hton_;
    int fd;//套接字
    struct sockaddr_in addr_to;//目标地址
    struct sockaddr_in addr_from;//本机地址
}target_;