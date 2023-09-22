#include <ros/ros.h> 
#include <serial/serial.h>  //ROS已经内置了的串口包 
#include <std_msgs/String.h>
// #include "odom/location_msg.h"
#include <string>
#include <vector>
#include <sstream>
#include <cmath>
#include <std_msgs/Int32.h>
#include <tf/transform_broadcaster.h>
#include <tf/transform_listener.h>
#include <nav_msgs/Odometry.h>
#include <sensor_msgs/Imu.h>
#include <geometry_msgs/Point.h>
#include <geometry_msgs/PoseWithCovarianceStamped.h>
#include <geometry_msgs/Twist.h>
#include <cstdlib>
#include <algorithm>
#include <random>
#define frequency 40
#define K_emst 0.0001
#define PI 3.141592654
#define fakedatatest 0

        serial::Serial ser;//C型版串口
        serial::Serial ser2;//主控串口
        bool serial_init();
        void location_callback(uint8_t* data);
        void set_odom(const geometry_msgs::PoseWithCovarianceStamped&);
        void TFpub(ros::Publisher &);
        // void TFpub();
        double intergrate_angle(double);
        void TFuodate(void*);
        void sendlocation();
        void modcallback(const std_msgs::Int32 &msg);
        std::string Odom_topic;

        ros::Time order_time;
        double dt;
        bool firstpos_flag=false;
        int mod=0;
        float xx,yy,yaya=0;//里程计读取的定位
        float x,y,yaw=0;//TF树获得的定位
        float v_xx,v_yy,v_yaya;
        float odom_init_x,odom_init_y,odom_init_yaw;
        
        // tf::Transform transform;
        tf::StampedTransform transform;
        tf::Quaternion odom_quaternion;
        std::string base_name,world_name;