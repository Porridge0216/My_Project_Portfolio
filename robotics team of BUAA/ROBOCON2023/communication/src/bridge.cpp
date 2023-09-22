#include <communication/bridge.hpp>

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

bool UDPhand()
{
    uint8_t data;
    uint16_t len = 1;
    int slen;
    data = 0x0f;
    slen=sendto(fd,&data,len,0,(struct sockaddr*)&addr_to,sizeof(addr_to)); 
    if(slen==-1)
    {
    printf("holdhand falure!\n");
    return false;
    }
    else
    {
    printf("holdhand successfully");
    return true;
    }
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
bool serialinit()
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
void serialread()
{
    //1.读取串口信息：
    //ROS_INFO_STREAM("Reading from serial port\n");
    //通过ROS串口对象读取串口信息
    static uint8_t buf[10240];
    int s;
    int n = ser.read(buf, ser.available());
    if(n!=0)std::cout << n << std::endl;
    for(int i = 0;i<n;i++)
    {
        s = (int)buf[i];
        std::cerr << std::hex << s << std::endl;
    }
    int len;
    if(n==0)return;
    len=sendto(fd,buf,n,0,(struct sockaddr*)&addr_to,sizeof(addr_to)); 
    if(len==-1){
    printf("send falure!\n");
    }
    else
    {
    printf("%d bytes have been sended successfully!\n",len);
    }
}
int main(int argc, char** argv)
{
    ros::init(argc, argv, "bridge_node");
    ros::NodeHandle nh;
    ros::Rate loop_rate(100);
    exam_msg.data="no_init";
    pthread_t thread_exam;
    int rc_exam = pthread_create(&thread_exam, NULL, exam_topic_publisher, NULL);
    std::cout<<"exam_topic pub"<<std::endl;

    nh.param<std::string>("to_ip",to_ip,"192.168.3.19");
    nh.param<int>("b_to_hton",to_hton,7777);

    nh.param<std::string>("from_ip",from_ip,"10.42.0.4");
    nh.param<int>("b_from_hton",from_hton,7777);

    while(!UDP_init())
    {
        loop_rate.sleep();
    }
    exam_msg.data="udp_init";
    serialinit();
    exam_msg.data="serial_init";
    pthread_t thread;

    int rc = pthread_create(&thread, NULL, UDPreviceve, NULL);
    while(ros::ok())
    {
    
    serialread();
    ros::spinOnce();
    loop_rate.sleep();
    }

    return 0;
}
