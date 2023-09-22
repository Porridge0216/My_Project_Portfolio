#include "communication/newpacket.hpp"


namespace newbr_packet{
    void newPacket::init(output_func output)
    {
        output_=output;
    }
    void newPacket::setPortCallback(port_func port_callback, int port){
        port_callback_[port] = port_callback;
    }
    void newPacket::receiveHanlder(uint8_t *data, uint16_t len){
        while (len > 0){
            switch (state_)
            {
            case 1:
                if (*data == 0x7E) state_ = 2;
                break;
            case 2:
                id_ = *data;
                state_ = 3;
                break;
            case 3:{
                sum = *data;
                state_ = 4;}
                break;
            case 4:
                port_ = *data;
                state_ = 5;
                break;
            case 5:
                type_ = *data;
                state_ = 6;
                break;
            case 6:
                data_len_ = *data;
                state_ = 7;
                break;
            case 7:
                if (cur_len_ < data_len_) {
                    recv_buff_[cur_len_] = *data; 
                    check_sum_ += *data;
                    ++cur_len_;
                }
                else if (check_sum_ == sum){
                    switch (type_)
                    {
                    case 0xa0:
                        type0Callback();
                        break;
                    case 0x20:
                        type1Callback();
                        break;
                    case 0x08:
                        type2Callback();
                    default:
                        break;
                    }
                    last_id_[level_] = id_;
                    reset();
                }
                else {
                    reset();
                }
            default:
                break;
            }
            --len;
            ++data;
        }
    }

    bool newPacket::sendData(uint8_t *data, uint16_t len, int type, int port, int level){
        uint8_t check_sum = 0;
        for (int i = 0; i < len; ++i){check_sum += data[i];}
        send_buff_[2] = check_sum;
        send_buff_[0] = 0x7E;
        send_buff_[3] = port;
        send_buff_[4] = type;
        send_buff_[5] = len;
        send_buff_[len+(uint16_t)FIX-1] = 0xE7;
        if (type == needreply){
            ++send_id_[level];
            send_buff_[1] = send_id_[level];
            
        }
        else send_buff_[1] = 0;
        memcpy(&send_buff_[6], data, len);
        if (type == needreply){
            if (len + (uint16_t)FIX > spare_len_[level]) return false;
            
            memcpy(buff_[level] + max_len_[level] - spare_len_[level], send_buff_, len + (uint16_t)FIX);
            spare_len_[level] -= len+ (uint16_t)FIX;
        }
        else output_(send_buff_, len + (uint16_t)FIX);
        return true; 
    }


    bool newPacket::update(){
        int flag = LEVEL_NUM;
        for (int i = LEVEL_NUM - 1; i >= 0; --i) 
        if (recv_flag_[i] == 0) flag = i;
        for (int i = LEVEL_NUM - 1; i >= 0; --i){
            float dt = getTime() - time_[i];
            if (((dt > overtime_ && recv_flag_[i] == 0) || i < flag) && (spare_len_[i] != max_len_[i])){
                uint16_t len = 0;
                ++retimes_[i];
                len = *(buff_[i]+5);
                len += (uint16_t)FIX;
                output_(buff_[i], len);
                time_[i] = getTime();
                std::cout << "i:" << i << std::endl;
                recv_flag_[i] = 0;
                if (retimes_[i] > 10) return true;
                else return false;
            }
        }
        return false;
    }

    float newPacket::getTime(){
        auto time_stt = clock();
        return (float)(time_stt / (float)myCLOCKS_PER_SEC);
    }

    void newPacket::type0Callback(){
        if (id_ != last_id_[level_]) {   
            port_callback_[port_](recv_buff_, data_len_);
            std::cout << "receive type 0 data" << std::endl;
        }
        else std::cout << "receive the same type0 data" << std::endl;
        recv_buff_[0] = id_;
        recv_buff_[1] = sum; 
        bool send_state = sendData(recv_buff_, 2, reply, port_, level_);
    }

    void newPacket::type1Callback(){
        port_callback_[port_](recv_buff_, data_len_);
    }

    void newPacket::type2Callback(){
        uint16_t data_len = 0;
        data_len = buff_[level_][5];
        memcpy(rece_buff_, buff_[level_], data_len);
        if (data_len_ != 2 || max_len_[level_] == spare_len_[level_] || memcmp(rece_buff_+1, recv_buff_, 2) != 0 ) return;
        retimes_[level_] = 0;
        data_len += (uint16_t)FIX;
        memcpy(buff_[level_], buff_[level_] + data_len, max_len_[level_] - spare_len_[level_] - data_len);
        spare_len_[level_] += data_len;
        recv_flag_[level_] = 1;        
        std::cout << "receive reply!" << std::endl;
    }

    void newPacket::reset() {
        state_ = 1;
        level_ = 0;
        type_ = 0;
        port_ = 0;
        check_sum_ = 0;
        cur_len_ = 0;
        data_len_ = 0;
        sum = 0;
    }
}
