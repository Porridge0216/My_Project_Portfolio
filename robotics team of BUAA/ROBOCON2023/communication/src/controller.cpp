#include <communication/controller.hpp>

bool UDPinit()
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

    image_addr_to.sin_family=AF_INET;
    image_addr_to.sin_port=htons(image_to_hton);
    image_addr_to.sin_addr.s_addr=inet_addr(cto_ip);

    addr_from.sin_family=AF_INET;
    addr_from.sin_port=htons(from_hton);//获得任意空闲端口
    addr_from.sin_addr.s_addr=inet_addr(cfrom_ip);//获得本机地址
    
    int len;
    len=bind(fd,(struct sockaddr*)&addr_from,sizeof(addr_from));
    if(len==-1)
    {
    printf("Bind error!\n");
    close(fd);
    return false;
    }
    printf("Bind successfully.\n");
    return true;
}

void* UDPreviceve(void* args)
{
    uint8_t buf[1024];
    socklen_t len;
    len = sizeof(sockaddr_in);
    ros::Rate loop_rate(60);
    unsigned long int ip;
    int recvnum = 0;
    while(ros::ok())
    {   
        static struct sockaddr_in from;
        recvnum=recvfrom(fd, buf, sizeof(buf),0, (struct sockaddr*)&from, &len);
        if(memcmp(&from.sin_addr.s_addr, &addr_to.sin_addr.s_addr, sizeof(addr_to.sin_addr.s_addr))!=0)
        {
            memcpy(&addr_to.sin_addr.s_addr,&from.sin_addr.s_addr,sizeof(addr_to.sin_addr.s_addr));
            std::cout << addr_to.sin_addr.s_addr<< std::endl;
        }
        //std::cerr << inet_ntoa(from.sin_addr) << std::endl;
        if(recvnum > 0)
        {
            std::cerr << recvnum << "bytes controller_data_receieve" << std::endl;
            // for(int i = 0;i<recvnum;i++){std::cerr << std::hex << (int)buf[i] << std::endl;}
            // std::cerr << "receieve_end" << std::endl;
            packet.receiveHanlder(buf, recvnum);
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

void square_mod_callback(uint8_t* data,uint16_t len){
    std_msgs::Int32 msg;
    float tempfloat;
    memcpy(&tempfloat,data,4);
    msg.data = (int) tempfloat;
    square_mod_pub.publish(msg);
    std::cerr << "mod_change: " << msg.data << std::endl;
    return;
}
void findblock_callback(uint8_t* data,uint16_t len)
{
    std_msgs::UInt8 msg;
    float tempfloat;
    memcpy(&tempfloat,data,4);
    msg.data = (uint8_t) tempfloat;
    findblock_pub.publish(msg);
    std::cerr << "findblock" << std::endl;
    image_need = true;
    return;
}
void findball_callback(uint8_t* data,uint16_t len)
{
    std_msgs::UInt8 msg;
    float tempfloat;
    memcpy(&tempfloat,data,4);
    msg.data = (uint8_t) tempfloat;
    findball_pub.publish(msg);
    std::cerr << "findball" << std::endl;
    return;
}
// void blockpos_callback(const find_cylinder::CylinderParam& msg)
// {
//     static uint8_t data[30];
//     static float tempbyte[6];
//     tempbyte[0] = msg.x;
//     memcpy(data,&tempbyte[0],4);
//     tempbyte[1] = msg.y;
//     memcpy(data+4,&tempbyte[1],4);
//     tempbyte[2] = msg.z;
//     memcpy(data+8,&tempbyte[2],4);
    
//     tempbyte[3] = msg.ox;
//     memcpy(data+12,&tempbyte[3],4);
//     tempbyte[4] = msg.oy;
//     memcpy(data+16,&tempbyte[4],4);
//     tempbyte[5] = msg.oz;
//     memcpy(data+20,&tempbyte[5],4);
//     packet.sendData(data,24,needreply,2,0);
//     std::cout << "blockpos_callback" << std::endl;
//     return;
    
// }


int main(int argc, char** argv)
{
    
    ros::init(argc, argv, "controller_node");
    ros::NodeHandle nh;
    ros::Rate loop_rate(50);
    //ip和端口读取
    nh.param<std::string>("to_ip",to_ip,"10.42.0.20");
    nh.param<int>("c_to_hton",to_hton,7777);
    nh.param<std::string>("from_ip",from_ip,"10.42.0.1");
    nh.param<int>("c_from_hton",from_hton,7777);
    nh.param<int>("image_to_hton", image_to_hton, 7777);

    //话题读取
    nh.param<std::string>("findblock_topic",findblock_topic,"findblock");
    nh.param<std::string>("findball_topic",findball_topic,"findball");
    nh.param<std::string>("blockpos_topic",blockpos_topic,"blockpos");
    nh.param<std::string>("ballpos_topic",ballpos_topic,"ballpos");
    nh.param<std::string>("image_topic",image_topic, "result_image");

    //参数读取

    findblock_pub = nh.advertise<std_msgs::UInt8>(findblock_topic,1);
    findball_pub = nh.advertise<std_msgs::UInt8>(findball_topic,1);
    // blockpos_sub = nh.subscribe(blockpos_topic,1,blockpos_callback);
    square_mod_pub = nh.advertise<std_msgs::Int32>("square_extraction_mode",2);
    //UDP初始化
    while(!UDPinit())
    {
        ros::spinOnce();
        loop_rate.sleep();
    }
    
    //协议初始化
    packet.init(UDPsend);
    packet.setPortCallback(findblock_callback,2);
    packet.setPortCallback(findball_callback,3);
    packet.setPortCallback(square_mod_callback,4);
    
    //接收线程
    pthread_t thread;
    int rc = pthread_create(&thread, NULL, UDPreviceve, NULL);



    while(ros::ok())
    {
        packet.update();
        ros::spinOnce();
        loop_rate.sleep();
    }
}
