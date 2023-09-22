#include <odom/odom.hpp>

bool serial_init()
{
    //初始化设置
    std::string Odom_topic;
    std::string vel_topic;

    

    //C型板串口设置
    restart: ROS_INFO_STREAM("Serial Port opening");
    try{   
    ser.setPort("/dev/ttyUSB1");
    ser.setBaudrate(115200);
    serial::Timeout to = serial::Timeout::simpleTimeout(1000);
    ser.setTimeout(to);
    ser.open();
    }
    catch (serial::IOException& e)
    {
        ROS_ERROR_STREAM("Unable to open Serial Port !");
        ros::Duration(0.25).sleep();
        goto restart;
        return false;
    }
    if (ser.isOpen())
    {
        ROS_INFO_STREAM("Serial Port initialized");
    }
    else
    {
        return false;
    }
    //主控串口设置
    // try{   
    // ser2.setPort("/dev/ttyUSB2");
    // ser2.setBaudrate(115200);
    // serial::Timeout to = serial::Timeout::simpleTimeout(1000);
    // ser2.setTimeout(to);
    // ser2.open();
    // }
    // catch (serial::IOException& e)
    // {
    //     ROS_ERROR_STREAM("Unable to open Serial Port !");
    //     return false;
    // }
    // if (ser.isOpen())
    // {
    //     ROS_INFO_STREAM("Serial Port initialized");
    //     return true;
    // }
    // else
    // {
    //     return false;
    // }

}

double intergrate_angle(double angle)
{
    //角度归一化
    while(angle>PI*2){angle-=PI*2;}
    while(angle<0){angle+=PI*2;}
    return angle;
}
void TFpub(ros::Publisher &pub)
{
    static tf::TransformBroadcaster tf_pub;
    //定位数据发布到TF树上
    transform.setIdentity();
    transform.setOrigin(tf::Vector3(xx, yy,0));
    odom_quaternion.setRPY(0,0,intergrate_angle(yaya));
    transform.setRotation(odom_quaternion);
    // transform.stamp_ = ros::Time::now();
    tf_pub.sendTransform(tf::StampedTransform(transform,ros::Time::now(),"odom","base_footprint"));
     nav_msgs::Odometry odom_msg;
     odom_msg.pose.pose.position.x = xx;
     odom_msg.pose.pose.position.y = yy;
     odom_msg.pose.pose.orientation = tf::createQuaternionMsgFromYaw(intergrate_angle(yaya));
     odom_msg.twist.twist.linear.x = v_xx;
     odom_msg.twist.twist.linear.y = v_yy;
     odom_msg.twist.twist.angular.z = v_yaya;
     odom_msg.header.frame_id = "base_footprint";
     odom_msg.header.stamp = ros::Time::now();
     pub.publish(odom_msg);
}
void location_callback(uint8_t* data)
{
    float tx,ty,ttheta;
    int int_tx,int_ty,int_ttheta;
    static int16_t Rxdata_16[6];
    memcpy(Rxdata_16, data, 12);
    tx = (float)Rxdata_16[0];
    tx/=1000;
    ty = (float)Rxdata_16[1];
    ty/=1000;
    ttheta = (float)Rxdata_16[2];
    ttheta/=10000;
    v_xx = (float)Rxdata_16[3];
    v_xx /= 1000;
    v_yy = (float)Rxdata_16[4];
    v_yy /= 1000;
    v_yaya = (float)Rxdata_16[5];
    v_yaya /= 1000;
    // xx = ty;yy=-tx;yaya = ttheta; //自己改了下
    xx = tx;yy = ty;yaya = ttheta;
    std::cout <<"pos"<< xx << " " << yy << " " << yaya <<std::endl;
}
// void sendlocation()
// {
//     static uint8_t odstart = 0x7f;
//     static uint8_t odtab = 0x11;
//     static uint8_t odlength = 0x0c;
//     static uint8_t checksum;
//     static uint8_t data[30] = {0};
//     static int16_t pos[3] = {0};
//     static int16_t vel[3] = {0};
    
//     checksum = 0x00;
//     data[0] = odstart;
//     data[1] = odtab;
//     data[2] = odlength;
//     // if(sqrt(x*x+y*y)<700)
//     // {
//     // order_time=ros::Time::now();
//     // order_time -= ros::Duration(7.);
//     // }
//     // if(ros::Time::now().sec-order_time.sec > 10)return;
    
//     pos[0] = (int) x;
//     pos[1] = (int) y;
//     pos[2] = (int) yaw;
//     vel[0] = (int) 0;
//     vel[1] = (int) 0;
//     vel[2] = (int) 0;
//     memcpy(data+3,&pos[0],2);
//     memcpy(data+5,&pos[1],2);
//     memcpy(data+7,&pos[2],2);
//     memcpy(data+9,&vel[0],2);
//     memcpy(data+11,&vel[1],2);
//     memcpy(data+13,&vel[2],2);
//     for(int j = 3;j < 15;++j)
//     {
//         checksum = checksum + data[j];
//     }
//     data[15] = checksum;
//     ser2.write(data,16);
//     std::cout <<"send"<< x <<" " << y << yaw << std::endl;
// }

// void* TFupdate(void* args)
// {
//     static tf::TransformListener tf_listener;
//     tf::StampedTransform base_world;
//     tf::StampedTransform base_odom;
//     tf::Quaternion Base_angle;
//     while(ros::ok())
// {if(mod ==0)
// {
//     try
//     {
//         if(!tf_listener.waitForTransform("/odom", "/square1",ros::Time(0), ros::Duration(0.05)))
//         {
//             ROS_ERROR("trans-Can not Wait Transform(square to base)");
//         }else
//             {
//             tf_listener.lookupTransform("odom", "square1",
//                                     ros::Time(0), base_world);
//             Base_angle = base_world.getRotation();
//             yaw = 0;
//             if(tf::getYaw(Base_angle)>PI)yaw =2*PI-tf::getYaw(Base_angle);
//             if(fabs(tf::getYaw(Base_angle)>0.6))yaw =tf::getYaw(Base_angle);
            
//             x = base_world.getOrigin().getX();
//             y = base_world.getOrigin().getY();
//             x = x*1000;
//             y = y*1000;
//             }
//     }
//     catch (tf::TransformException &ex)
//     {
//       ROS_ERROR("%s",ex.what());
//       ros::Duration(1.0).sleep();
//     }
// }else
// {
//     try
//     {
//         if(!tf_listener.waitForTransform("/odom", "/square2",ros::Time(0), ros::Duration(0.05)))
//         {
//             ROS_ERROR("trans-Can not Wait Transform(square to base)");
//         }else
//             {
//             base_world.stamp_ = ros::Time::now();
//             tf_listener.lookupTransform("odom", "square2",
//                                     ros::Time(0), base_world);
//             Base_angle = base_world.getRotation();
//             yaw = 0;
//             if(tf::getYaw(Base_angle)>PI)yaw =2*PI-tf::getYaw(Base_angle);
//             if(fabs(tf::getYaw(Base_angle)>0.6))yaw =tf::getYaw(Base_angle);
//             yaw =tf::getYaw(Base_angle);
//             x = base_world.getOrigin().getX();
//             y = base_world.getOrigin().getY();
//             x = x*1000;
//             y = y*1000;
//             yaw = yaw*10000;
//             }
//     }
//     catch (tf::TransformException &ex)
//     {
    
//       ROS_ERROR("%s",ex.what());
//       ros::Duration(1.0).sleep();
//     }
// }

// }

// return 0;
// }

void modcallback(const std_msgs::Int32 &msg)
{
    mod = msg.data;
    order_time = ros::Time::now();
}
int main(int argc, char** argv)
{
    ros::init(argc, argv, "odom_node");
    ros::NodeHandle nh;
    ros::Publisher Odom_pub;
    ros::Subscriber mod_sub;
    nh.param<std::string>("/base_link", base_name, "base_link");
    nh.param<std::string>("/world", world_name, "world");
    nh.param<std::string>("Odom_topic",Odom_topic,"odom");
    Odom_pub = nh.advertise<nav_msgs::Odometry>(Odom_topic,1);
    mod_sub = nh.subscribe("square_extraction_mode",2,modcallback);
    order_time = ros::Time::now();
    //初始化里程计
    nh.param<float>("/odom_init_x",odom_init_x,0);
    nh.param<float>("/odom_init_y",odom_init_y,0);
    nh.param<float>("/odom_init_yaw",odom_init_yaw,0);
    bool odom_reflag;
    ros::Rate loop_rate(frequency);
    main_restart:
    serial_init();
    pthread_t thread;
    // int rc = pthread_create(&thread, NULL, TFupdate, NULL);
    
    uint8_t buffer[1024];
    uint8_t buffer2[1024];
    uint8_t odstart = 0x7f;
    uint8_t odtab = 0x11;
    uint8_t odlength = 0x0c;
    while (ros::ok()) //串口读取里程计信息
    {
 
        try{
        if (ser.available())
        {
            //1.读取串口信息：
            //通过ROS串口对象读取串口信息
            
            int n = ser.read(buffer, ser.available());
            //2.截取数据、解析数据：

            uint8_t* data;
            uint8_t sum = 0;
            int i = 0, start = -1, end = -1;
            for(i=0;i<n-15;i++)
            {
                sum = 0;
                if(buffer[i] == odstart && buffer[i+1] == odtab && buffer[i+2] == odlength)
                    {
                        data = &buffer[i+3];
                        for(int j=0;j<12;j++)
                        {
                            sum+= *(data+j);
                        }
                        if(sum!=*(data+12))continue;
                        location_callback(data);
                        
                    break;
                    }
            }
        }
        }
        catch (serial::IOException& e)
        {
            goto main_restart;
        }
        // if(ser2.available())
        // {
        //     int len = ser2.read(buffer2, ser2.available());
        //     ser.write(buffer2,len);
        // }

        // sendlocation();
        TFpub(Odom_pub); //发布到tf树上？
        ros::spinOnce();
        loop_rate.sleep();
        
    }
    
    return 0;
}
