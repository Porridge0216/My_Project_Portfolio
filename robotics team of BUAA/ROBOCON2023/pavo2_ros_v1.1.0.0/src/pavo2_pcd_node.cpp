#include <ros/ros.h>
#include <sensor_msgs/PointCloud.h>
#include "pavo2_driver.h"
#define MAX_RESPONSES (4096)
using namespace pavo2;
pavo2_driver *drv = NULL;
int main(int argc, char** argv){
  ros::init(argc, argv, "point_cloud_publisher");
  std::string frame_id, lidar_ip, host_ip, cloud_topic;
  int lidar_port, host_port;
  int angle_resolution,motor_speed;
  int angle_start,angle_end;
  double angle_min;
  double angle_max;
  bool inverted,enable_motor,intensity;
  ros::NodeHandle nh;
  
  ros::NodeHandle nh_private("~");
  nh_private.param<std::string>("frame_id", frame_id, "pcd_frame");
  nh_private.param<std::string>("cloud_topic",cloud_topic,"cloud");
  nh_private.param<double>("angle_max", angle_max , 140.0);
  nh_private.param<double>("angle_min", angle_min , -140.0);
  nh_private.param<int>("angle_start", angle_start, 4000);
  nh_private.param<int>("angle_end", angle_end, 32000);
  nh_private.param<bool>("intensity", intensity, true);
  nh_private.param<bool>("inverted", inverted,false );
  nh_private.param<int>("motor_speed", motor_speed, 50);
  nh_private.param<int>("angle_resolution", angle_resolution, 32);
  
  nh_private.param<std::string>("lidar_ip", lidar_ip , "10.10.10.121");
  nh_private.param<int>("lidar_port", lidar_port , 2368);
  nh_private.param<std::string>("host_ip", host_ip ,"10.10.10.100");
  nh_private.param<int>("host_port", host_port , 2368);
 

  ros::Publisher cloud_pub = nh.advertise<sensor_msgs::PointCloud>(cloud_topic, 1000);

  std::vector<pavo_response_pcd_t> pcd_vec;
  int num_points;
  drv=new pavo2_driver();
  if(NULL==drv)
  {
    ROS_INFO("new driver err");
  }
  ROS_INFO("IP:%s Port: %d",lidar_ip.c_str(),lidar_port);
  if(drv->pavo2_open(lidar_ip, lidar_port))
  {
    ROS_INFO("open socket succ");
  }
  else
  {
    ROS_INFO("open socket err");
  }
  uint8_t motor_speed_get=0;
  drv->get_motor_speed(motor_speed_get);
  sleep(1);
  
  ROS_INFO("laser speed %d ",motor_speed_get);
  //drv->set_motor_speed(motor_speed);

  ROS_INFO("angle_resolution: %d",angle_resolution);
  drv->set_angle_resolution(angle_resolution);
   
  ROS_INFO("angle_start:%d angle_end: %d",angle_start,angle_end);
  drv->set_angle_range(angle_start,angle_end);

  ROS_INFO("intensity:%d ",intensity);
  drv->set_intensity_mode(intensity);
  sleep(1);
  
  drv->enable_data(true);
  ros::Rate rate(motor_speed);
  int count = 0;
  while(nh.ok()){
    sensor_msgs::PointCloud cloud;
    drv->get_scanned_data(pcd_vec,150);
    num_points = pcd_vec.size();
//     ROS_INFO(" num_points:%d ", num_points);

    cloud.header.stamp = ros::Time::now();
    cloud.header.frame_id = frame_id;
 
    cloud.points.resize(num_points);

    //add an intensity channel to the cloud
    cloud.channels.resize(1);
    cloud.channels[0].name = "intensities";
    cloud.channels[0].values.resize(num_points);
        if (!inverted)
        {
              
                for(unsigned int i = 0; i < num_points; i++){
                        cloud.points[i].x = -pcd_vec[i].x * 0.002f ;
                        cloud.points[i].y = -pcd_vec[i].y * 0.002f ;
                        cloud.points[i].z = pcd_vec[i].z;
                        cloud.channels[0].values[i] = pcd_vec[i].intensity;
                }       
        }
        else 
        {
                
                for(unsigned int i = 0; i < num_points; i++){
                        cloud.points[num_points-1-i].x = -pcd_vec[i].x * 0.002f ;
                        cloud.points[num_points-1-i].y = pcd_vec[i].y * 0.002f ;
                        cloud.points[num_points-1-i].z = pcd_vec[i].z;
                        cloud.channels[0].values[num_points-1-i] = pcd_vec[i].intensity;                 
                }
        }
        
    cloud_pub.publish(cloud);
    rate.sleep();
  }

    delete drv;

}
