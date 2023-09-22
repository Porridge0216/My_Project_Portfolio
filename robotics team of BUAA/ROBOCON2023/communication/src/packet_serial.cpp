#include "communication/packet_serial.hpp"


namespace br_packet{
    void Packet::init(output_func output)
    {
        output_=output;
    }

    void Packet::setPortCallback(port_func port_callback, int port){
        port_callback_[port] = port_callback;
    }

    void Packet::receiveHanlder(uint8_t *data, uint16_t len){
        while (len > 0){
            switch (state_)
            {
            case 1:
                if (*data == 0xff) state_ = 2;
                break;
            case 2:
                data_len_ |= *data;
                data_len_ << 8;
                state_ = 3;
                break;
            case 3:
                data_len_ |= *data;
                state_ = 4;
                break;
            case 4:
                type_ = (*data) >> 6;
                port_ = (*data) & 0x0F;
                level_ = ((*data) >> 4) & 0x03;
                state_ = 5;
                break;
            case 5:
                id_ = *data;
                state_ = 6;
                break;
            case 6:
                if (cur_len_ < data_len_) {
                    recv_buff_[cur_len_] = *data; 
                    check_sum_ += *data;
                    ++cur_len_;
                }
                else if (check_sum_ == *data){
                    switch (type_)
                    {
                    case 0:
                        type0Callback();
                        break;
                    case 1:
                        type1Callback();
                        break;
                    case 2:
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

    bool Packet:: sendData(uint8_t *data, uint16_t len, int type, int port, int level){
        uint8_t check_sum = 0;
        for (int i = 0; i < len; ++i){check_sum += data[i];}
        int ilen = len;
        send_buff_[ilen + 5] = check_sum;
        send_buff_[0] = 0xFF;
        send_buff_[1] = (uint8_t)(len >> 8);
        send_buff_[2] = (uint8_t)(len & 0xFF);
        send_buff_[3] = (uint8_t)((type << 6) | port | (level << 4));
        if (type == 0){
            ++send_id_[level];
            send_buff_[4] = send_id_[level];
        }
        else send_buff_[4] = 0;
        memcpy(&send_buff_[5], data, len);
        
        if (type == 0){
            if (len + (uint16_t)FIX > spare_len_[level]) return false;
            memcpy(buff_[level] + max_len_[level] - spare_len_[level], send_buff_, len + (uint16_t)FIX);
            spare_len_[level] -= len+ (uint16_t)FIX;
        }
        else   output_(send_buff_, len + (uint16_t)FIX);
        return true;

    }


    bool Packet::update(){
        int flag = LEVEL_NUM;
        for (int i = LEVEL_NUM - 1; i >= 0; --i) 
        if (recv_flag_[i] == 0) flag = i;
        for (int i = LEVEL_NUM - 1; i >= 0; --i){
            double dt = getTime() - time_[i];
            // std::cout <<"dt"<< dt << "over_time "<< overtime_ <<std::endl;
            if (((dt > overtime_ && recv_flag_[i] == 0) || i < flag) && (spare_len_[i] != max_len_[i])){
                uint16_t len = 0;
                ++retimes_[i];
                len |= buff_[i][1];
                len <<= 8;
                len |= buff_[i][2];
                len += (uint16_t)FIX;
                output_(buff_[i], len);
                time_[i] = getTime();
                recv_flag_[i] = 0;
                if (retimes_[i] > 10) return true;
                else return false;
            }
        }
        return false;
    }

    float Packet::getTime(){
        auto time_stt = clock();
        return time_stt / (float)CLOCKS_PER_SEC;
        // static float clock_per_sec = 100000000;
        // return time_stt / (float) clock_per_sec;
    }

    void Packet::type0Callback(){
        if (id_ != last_id_[level_]) {   
            port_callback_[port_](recv_buff_, data_len_);
            std::cout << "receive type 0 data" << std::endl;
        }
        else std::cout << "receive the same type0 data" << std::endl;
        bool send_state = sendData(recv_buff_, data_len_, 2, 0, level_);
    }

    void Packet::type1Callback(){
        port_callback_[port_](recv_buff_, data_len_);

    }

    void Packet::type2Callback(){
        uint16_t data_len = 0;
        data_len |= buff_[level_][1];
        data_len << 8;
        data_len |= buff_[level_][2];
        memcpy(send_buff_, buff_[level_] + 5, data_len);
        if (data_len != data_len_ || max_len_[level_] == spare_len_[level_] || memcmp(send_buff_, recv_buff_, data_len) != 0) return;
        retimes_[level_] = 0;
        data_len += (uint16_t)FIX;
        memcpy(buff_[level_], buff_[level_] + data_len, max_len_[level_] - spare_len_[level_] - data_len);
        spare_len_[level_] += data_len;
        recv_flag_[level_] = 1;        
        std::cout << "receive reply!" << std::endl;
    }

    void Packet::reset() {
        state_ = 1;
        level_ = 0;
        type_ = 0;
        port_ = 0;
        check_sum_ = 0;
        cur_len_ = 0;
        data_len_ = 0;
    }
}
