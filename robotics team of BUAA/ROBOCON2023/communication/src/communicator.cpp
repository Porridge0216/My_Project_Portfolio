#include <communication/communicator.hpp>
Communicator::Communicator(ros::NodeHandle &nh, ros::NodeHandle &nh_local):
nh_(nh),nh_local_(nh_local)
{
    loadParameters();
    UDPinit();
    robotPacket.init(UDPsend,&robot);
    controllerPacket.init(UDPsend,&controller);
    robotPacket.setPortCallback(example_robot_port1_Callback,1);
    controllerPacket.setPortCallback(controller_port3_Callback,3);
    controllerPacket.setPortCallback(controller_port2_Callback,2);
    controllerPacket.setPortCallback(set_field_Callback,1);
    controllerPacket.setPortCallback(example_controller_port0_Callback,0);
    std::thread work_thread1(robotUDPrece, &robot,&robotPacket);
    work_thread1.detach();
    std::thread work_thread2(controllerUDPrece, &controller,&controllerPacket);
    work_thread2.detach();
    robot.Name_ = "robot";
    controller.Name_ = "controller";
    
    
    rosInit();
}

void Communicator::loadParameters()
{
    nh_local_.param<std::string>("/robot_ip",robot.to_ip_,"10.42.0.1");
    ROS_DEBUG("robot_ip:%s",robot.to_ip_.c_str());
    nh_local_.param<std::string>("/controller_ip",controller.to_ip_,"10.42.0.40");
    ROS_DEBUG("controller_ip:%s",controller.to_ip_.c_str());

    std::string from_ip;
    nh_local_.param<std::string>("/from_ip",from_ip,"10.42.0.4");
    ROS_DEBUG("from_ip:%s",from_ip.c_str());
    robot.from_ip_ = from_ip;
    controller.from_ip_ = from_ip;

    nh_local_.param<int>("/robot_to_hton",robot.to_hton_,1347);
    ROS_DEBUG("robot_to_hton: %d",robot.to_hton_);
    nh_local_.param<int>("/robot_from_hton",robot.from_hton_,7777);
    ROS_DEBUG("robot_from_hton: %d",robot.from_hton_);
    nh_local_.param<int>("/controller_to_hton",controller.to_hton_,6666);
    ROS_DEBUG("controller_to_hton: %d",controller.to_hton_);
    nh_local_.param<int>("/controller_from_hton",controller.from_hton_,6666);
    ROS_DEBUG("controller_from_hton: %d",controller.from_hton_);

    nh_local_.param<std::string>("/example_pub_topic",example_pub_topic,"examplepub");
    ROS_DEBUG("example_pub_topic:%s",example_pub_topic.c_str());
    nh_local_.param<std::string>("/port2_sub_topic",port2_sub_topic,"/rings_detect/result");//约定topic
    ROS_DEBUG("port2_sub_topic:%s",port2_sub_topic.c_str());
    nh_local_.param<std::string>("/port3_pub_topic",port3_pub_topic,"/need_shoot_aid");//约定topic
    ROS_DEBUG("port3_pub_topic:%s",port3_pub_topic.c_str());
    nh_local_.param<std::string>("/set_field_srv",set_field_srv,"/set_field");//约定topic
    ROS_DEBUG("set_field_srv:%s",set_field_srv.c_str());

}
void Communicator::rosInit()
{
    //subscribe
    // example_sub = nh_.subscribe(example_sub_topic,10,&Communicator::example_ros_Callback,this);
    port2_sub = nh_.subscribe(port2_sub_topic,10,&Communicator::port2_sub_ros_Callback,this);

    //advertise
    example_pub = nh_.advertise<std_msgs::UInt8>(example_pub_topic,10);
    port2_pub = nh_.advertise<std_msgs::UInt8>(port2_pub_topic,10);
    port3_pub = nh_.advertise<std_msgs::UInt8>(port3_pub_topic,10);
    set_field_client = nh_.serviceClient<communication::set_field>(set_field_srv);
}
void Communicator::UDPinit()
{
    const char *robot_to_ip = robot.to_ip_.c_str();
    const char *robot_from_ip = robot.from_ip_.c_str();
    const char *controller_to_ip = controller.to_ip_.c_str();
    const char *controller_from_ip = controller.from_ip_.c_str();
    
    controller.addr_to.sin_family=AF_INET;
    controller.addr_to.sin_port=htons(controller.to_hton_);
    controller.addr_to.sin_addr.s_addr=inet_addr(controller_to_ip);

    controller.addr_from.sin_family=AF_INET;
    controller.addr_from.sin_port=htons(controller.from_hton_);//获得任意空闲端口
    controller.addr_from.sin_addr.s_addr=inet_addr(controller_from_ip);//获得本机地s址

    robot.addr_to.sin_family=AF_INET;
    robot.addr_to.sin_port=htons(robot.to_hton_);
    robot.addr_to.sin_addr.s_addr=inet_addr(robot_to_ip);

    robot.addr_from.sin_family=AF_INET;
    robot.addr_from.sin_port=htons(robot.from_hton_);//获得任意空闲端口
    robot.addr_from.sin_addr.s_addr=inet_addr(robot_from_ip);//获得本机地s址
    
    robot.fd=socket(AF_INET,SOCK_DGRAM,0);
    controller.fd=socket(AF_INET,SOCK_DGRAM,0);

    int len = 0;
    do
    {
        if(len==-1)close(controller.fd);
        printf("controller Bind!\n");
        len=bind(controller.fd,(struct sockaddr*)&controller.addr_from,sizeof(controller.addr_from));
    }while(len==-1);
    printf("controller Bind successfully.\n");
    do
    {
        if(len==-1)close(robot.fd);
        printf("robot Bind!\n");
        len=bind(robot.fd,(struct sockaddr*)&robot.addr_from,sizeof(robot.addr_from));
    }while(len==-1);
    printf("robot Bind successfully.\n");
}
void Communicator::robotUDPrece(target* t_adr,br_packet::Packet *P_adr)
{
    uint8_t buf[1024];
    socklen_t len;
    len = sizeof(sockaddr_in);//逻辑(一秒10次)
    ros::Rate loop_rate(60);
    int recvnum = 0;
    while(ros::ok())
    {   
        static struct sockaddr_in from;
        recvnum=recvfrom(t_adr->fd, buf, sizeof(buf),0, (struct sockaddr*)&from, &len);
        if(memcmp(&from.sin_addr.s_addr, &(t_adr->addr_to.sin_addr.s_addr), sizeof(t_adr->addr_to.sin_addr.s_addr))!=0)
        {
            memcpy(&t_adr->addr_to.sin_addr.s_addr,&from.sin_addr.s_addr,sizeof(t_adr->addr_to.sin_addr.s_addr));
            std::cerr << t_adr->Name_ <<"ipchanged: "<< inet_ntoa(from.sin_addr) << std::endl;
        }
        if(recvnum > 0)
        {
            std::cerr << recvnum << "bytes _data_receieve" << std::endl;
            P_adr->receiveHanlder(buf, recvnum);
        }
        loop_rate.sleep();
    }
}  
void Communicator::controllerUDPrece(target* t_adr,br_packet::Packet *P_adr)
{
    uint8_t buf[1024];
    socklen_t len;
    len = sizeof(sockaddr_in);
    ros::Rate loop_rate(60);
    int recvnum = 0;
    while(ros::ok())
    {   
        static struct sockaddr_in from;
        recvnum=recvfrom(t_adr->fd, buf, sizeof(buf),0, (struct sockaddr*)&from, &len);
        if(memcmp(&from.sin_addr.s_addr, &(t_adr->addr_to.sin_addr.s_addr), sizeof(t_adr->addr_to.sin_addr.s_addr))!=0)
        {
            memcpy(&t_adr->addr_to.sin_addr.s_addr,&from.sin_addr.s_addr,sizeof(t_adr->addr_to.sin_addr.s_addr));
            std::cerr << t_adr->Name_ <<"ipchanged: "<< inet_ntoa(from.sin_addr) << std::endl;
        }
        if(recvnum > 0)
        {
            std::cerr << recvnum << "bytes controller_data_receieve" << std::endl;
            P_adr->receiveHanlder(buf, recvnum);
        }
        loop_rate.sleep();
    }
}                             
void Communicator::UDPsend(uint8_t* data, uint16_t len, target* t_adr)
{
    uint8_t buf[1024];
    int slen;
    slen=sendto(t_adr->fd,data,len,MSG_DONTWAIT,(struct sockaddr*)&t_adr->addr_to,sizeof(t_adr->addr_to)); 
    if(slen==-1)
    {
    printf("%s send falure!\n", t_adr->Name_.c_str());
    }
    else
    {
    printf("%d bytes have been sended successfully by %s!\n",slen, t_adr->Name_.c_str());
    }
}
void Communicator::example_ros_Callback(const std_msgs::UInt8& msg)
{
    static uint8_t data[5];
    data[0]=msg.data;
    std::cout<<(int)data[0]<<std::endl;
    // robotPacket.sendData(data,2,0x01,1,0);
    controllerPacket.sendData(data,2,0x01,0,0);
    return;
}

void Communicator::example_robot_port1_Callback(uint8_t* data,uint16_t len)
{
    std_msgs::UInt8 msg;
    int tempint;
    memcpy(&tempint,data,4);
    std::cout<<tempint<<std::endl;
    return;
}

void Communicator::set_field_Callback(uint8_t* data,uint16_t len)
{
    float tempfloat[3];
    memcpy(tempfloat,data,4);
    memcpy(tempfloat+1,data+4,4);
    memcpy(tempfloat+2,data+8,4);
    communication::set_field set_field_service;
    set_field_service.request.x = tempfloat[0];
    set_field_service.request.y = tempfloat[1];
    set_field_service.request.theta = tempfloat[2];
    bool flag=set_field_client.call(set_field_service);
    if(flag)
    std::cout<<"set_field service called"<<std::endl;
    return;
}

void Communicator::example_controller_port0_Callback(uint8_t* data, uint16_t len)
{
    std_msgs::UInt8 msg;
    float tempfloat;
    memcpy(&tempfloat,data,4);
    msg.data = (uint8_t) tempfloat;
    example_pub.publish(msg);
    std::cerr << "port0_msg_received" << std::endl;
    return;
}

void Communicator::controller_port2_Callback(uint8_t* data, uint16_t len)
{
    std_msgs::UInt8 msg;
    bool tempbool;
    memcpy(&tempbool,data,1);
    msg.data = (uint8_t) tempbool;
    port2_pub.publish(msg);
    std::cerr << "port2_msg_received" << std::endl;
    return;
}

void Communicator::port2_sub_ros_Callback(const communication::rings& msg)
{
    static uint8_t data[10];
    static float tempfloat[2];
    if(msg.isempty)
    return;
    else
    {
        tempfloat[0]=msg.data_list[0].x;
        memcpy(data,tempfloat,4);
        tempfloat[1]=msg.data_list[0].y;
        memcpy(data+4,tempfloat+1,4);
        controllerPacket.sendData(data,8,pcdata,2,0);
        std::cout << "port2_callback" << std::endl;
        return;
    }
}

void Communicator::controller_port3_Callback(uint8_t* data,uint16_t len)
{
    std_msgs::UInt8 msg;
    uint8_t tempuint8;
    memcpy(&tempuint8,data,1);
    msg.data = (uint8_t) tempuint8;
    port3_pub.publish(msg);
    std::cerr << "port3_msg_received" << std::endl;
    return;
}
