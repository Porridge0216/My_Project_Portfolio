#include <communication/trans_scm.hpp>
bool UDP_init()
{
    std::cerr << "udp_init" << std::endl;
    fd=socket(AF_INET,SOCK_DGRAM,0);
    if(fd==-1)
    {
    perror("socket create error!\n");
    return false;
    }
    printf("socket fd=%d\n",fd);

    const char *cto_ip = to_ip.c_str();
    const char *cfrom_ip = from_ip.c_str();
    addr_to.sin_family=AF_INET;
    addr_to.sin_port=htons(to_hton);
    addr_to.sin_addr.s_addr=inet_addr(cto_ip);
    addr_from.sin_family=AF_INET;
    addr_from.sin_port=htons(from_hton);//获得任意空闲端口
    addr_from.sin_addr.s_addr=inet_addr(cfrom_ip);//获得本机地址

    r=bind(fd,(struct sockaddr*)&addr_from,sizeof(addr_from));
    if(r==-1)
    {
    close(fd);
    return false;
    }
    return true;
}



void* UDPreviceve(void* args)
{ 
    static uint8_t buf[10240];
    socklen_t len;
    len = sizeof(sockaddr_in);
    ros::Rate loop_rate(150);
    unsigned long int ip;
    int recvnum = 0;
    while(ros::ok())
    {   
        static struct sockaddr_in from;
        recvnum=recvfrom(fd, buf, sizeof(buf),0, (struct sockaddr*)&from, &len);
        if(memcmp(&from.sin_addr.s_addr, &addr_to.sin_addr.s_addr, sizeof(addr_to.sin_addr.s_addr))!=0)
        {
            memcpy(&addr_to.sin_addr,&from.sin_addr,sizeof(addr_to.sin_addr));
        }
        //std::cerr << inet_ntoa(from.sin_addr) << std::endl;
        if(recvnum > 0)
        {
            // std::cerr << "receieve" << std::endl;
            // for(int i = 0;i<len;i++){std::cerr << std::hex << (int)buf[i] << std::endl;}
            // std::cerr << "receieve" << std::endl;
            ser.write(buf,recvnum);
        }
        ros::spinOnce();
        loop_rate.sleep();
    }
}

void UDPsend(uint8_t* data, uint16_t len)
{
    uint8_t buf[1024];
    int slen;
    slen=sendto(fd,data,len,0,(struct sockaddr*)&addr_to,sizeof(addr_to)); 
    //std::cerr << inet_ntoa(addr_to.sin_addr) <<std::endl;
    //for(int i = 0;i<slen;i++){std::cerr << std::hex << (int)data[i] << std::endl;}
    if(slen==-1)
    {
    printf("send falure!\n");
    }
    else
    {
    printf("%d bytes have been sended successfully!\n",slen);
    }
}

bool serial_init()
{
    try{   
    ser.setPort("/dev/ttyUSB0");
    ser.setBaudrate(115200);
    serial::Timeout to = serial::Timeout::simpleTimeout(1000);
    ser.setTimeout(to);
    ser.open();
    }
    catch (serial::IOException& e)
    {
        std::cout << "no serial" << std::endl;
        ROS_ERROR_STREAM("Unable to open Serial Port !");
        return false;
    }
    if (ser.isOpen())
    {
        ROS_INFO_STREAM("Serial Port initialized");
        return true;
    }
    else
    {
        return false;
    }
}

void write(uint8_t* buff, uint16_t len)
{
    int ilen = len;
    // int dataa;
    // for (int i =0 ;i<ilen;i++)
    // {
    //     dataa = *(buff+i);
    //     std::cout << std::hex << dataa << std::endl;
    // }
    ser.write(buff, len);
    // try
    // {
    //     ser.write(buff, len);
    // }
    // catch(serial::PortNotOpenedException& e)
    // {
    //     ROS_ERROR_STREAM("Serial unfortunately close");
    // }
}


// void location_callback(uint8_t* data, uint16_t len){
//     static float pos[3];
//     memcpy(&pos[0],data,12);

//     base_world.setIdentity();
//     base_world.setOrigin(tf::Vector3(pos[0], pos[1], 0.0)); 
//     q.setRPY(0, 0, pos[2]);
//     base_world.setRotation(q);
// }



void pos_callback(const nav_msgs::Odometry& msg){
    
    static uint8_t data[20];
    float tempfloat[3];
    tempfloat[0]=msg.pose.pose.position.x;
    tempfloat[1]=msg.pose.pose.position.y;
    tempfloat[2]=tf::getYaw(msg.pose.pose.orientation);
    memcpy(data,tempfloat,4);
    memcpy(data+4,tempfloat+1,4);
    memcpy(data+8,tempfloat+2,4);
    packet.sendData(data,12,pcdata,2,0);
    std::cout<<"compensation sent"<<std::endl;

}

void img_angle_callback(uint8_t* data,uint16_t len){
    communication::head_angle msg;
    float tempfloat;
    memcpy(&tempfloat,data,4);
    msg.angle =  tempfloat;
    img_pub.publish(msg); 
    std::cout << "head_angle_msg_received" << std::endl;
    return;
}

void shoot_aid_sub_callback(const std_msgs::UInt8& msg)
{
    static uint8_t data[5];
    float tempfloat;
    uint8_t tempuint8=msg.data;
    communication::shoot_aid aid_service;
    aid_service.request.target_id=tempuint8;
    bool flag=shoot_aid_client.call(aid_service);
    if(flag)
    {
        tempfloat=aid_service.response.offset;
        std::cout<<tempfloat<<std::endl;
        memcpy(data,&tempfloat,4);
        packet.sendData(data,4,pcdata,1,0);
    }
}

void port0_callback(uint8_t* data,uint16_t len){
    ROS_INFO_STREAM("bridge data sended");
    return;
}

void* exam_topic_publisher(void* args)
{
    ros::Rate loop_rate(100);
    ros::NodeHandle nh;
    ros::Publisher exam_pub = nh.advertise<std_msgs::String>("exam_topic",1);
    while(ros::ok())
    {
        exam_pub.publish(exam_msg);
        ros::spinOnce();
        loop_rate.sleep();
    }
    
}

// void* TFpub(void* args)
// {

//     tf_pub.sendTransform(tf::StampedTransform(base_world, 
//                            ros::Time::now(),
//                            world_name, base_name));
// }

int main(int argc, char** argv)
{
    ros::init(argc, argv, "trans_node");
    ros::NodeHandle nh;
    ros::Rate loop_rate(100);
    exam_msg.data="no_init";
    pthread_t thread_exam;
    int rc_exam = pthread_create(&thread_exam, NULL, exam_topic_publisher, NULL);

   
    packet.init(write);
    
    nh.param<std::string>("/base_link", base_name, "base_link");
    nh.param<std::string>("/world", world_name, "world");
    nh.param<std::string>("img_angle_topic",img_angle_topic,"/head_angle");
    nh.param<std::string>("pos_topic",pos_topic,"/compensation");
    nh.param<std::string>("shoot_aid_srv",shoot_aid_srv,"/shoot_aid");
    nh.param<std::string>("shoot_aid_topic",shoot_aid_topic,"/need_shoot_aid");
    
    nh.param<std::string>("to_ip",to_ip,"192.168.3.19");
    nh.param<int>("b_to_hton",to_hton,7777);

    nh.param<std::string>("from_ip",from_ip,"10.42.0.4");
    nh.param<int>("b_from_hton",from_hton,7777);

    packet.setPortCallback(port0_callback,0);
    packet.setPortCallback(img_angle_callback,1);
    // base_name = "base_link";
    // world_name = "world";
    // pthread_t thread;
    // int rc = pthread_create(&thread, NULL, TFpub, NULL);
    img_pub = nh.advertise<communication::head_angle>(img_angle_topic,1);
    shoot_aid_sub = nh.subscribe(shoot_aid_topic,1,&shoot_aid_sub_callback);
    pos_sub = nh.subscribe(pos_topic,1,&pos_callback);
    shoot_aid_client = nh.serviceClient<communication::shoot_aid>(shoot_aid_srv);
    
    serial_restart: ROS_INFO_STREAM("serial opening");
    while(!UDP_init())
    {
        loop_rate.sleep();
    }
    exam_msg.data="udp_init";
    serial_init();
    exam_msg.data="serial_init";
    pthread_t thread;

    int rc = pthread_create(&thread, NULL, UDPreviceve, NULL);
    // ros::service::waitForService(shoot_aid_srv);
    int templen;
    while(ros::ok())
    {
       if(ser.isOpen())
        {
            if(ser.available())
        {
        len = ser.read(buff, ser.available());
        templen=sendto(fd,buff,len,0,(struct sockaddr*)&addr_to,sizeof(addr_to)); 
        // if(templen==-1){
        // printf("send falure!\n");
        // }
        // else
        // {
        // printf("%d bytes have been sended successfully!\n",len);
        // }
        
        packet.receiveHanlder(buff, len);
        packet.update();}
        }
        else
        {
            exam_msg.data="serial err";
            goto serial_restart;
        }
        

        ros::spinOnce();
        loop_rate.sleep();
    
    }
}
