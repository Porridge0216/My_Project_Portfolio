#ifndef BR_PACKET
#define BR_PACKET

#include <deque>
#include <array>
#include <vector>
#include <stdint.h>
#include <ctime>
#include <cstring>
#include <iostream>

#define PORT_NUM 16
#define LEVEL_NUM 4
#define BUFF_SIZE 200
#define FIX 6

namespace br_packet{
    typedef void(*port_func)(uint8_t*, uint16_t);
    typedef void(*output_func)(uint8_t*, uint16_t);


    class Packet{
        public:
            Packet(float overtime, output_func output): overtime_(overtime), output_(output) {
                buff_[0] = level0_buff_;
                buff_[1] = level1_buff_;
                buff_[2] = level2_buff_;
                buff_[3] = level3_buff_;
            }
            Packet(output_func output): Packet(0.001, output) {}
            Packet(const Packet& pac) = default;
            Packet(): Packet(0.1, nullptr) {}
            void init(output_func);
            void setPortCallback(port_func port_callback, int port);
            void receiveHanlder(uint8_t *data, uint16_t len);
            bool sendData(uint8_t *data, uint16_t len, int type, int port, int level);
            bool update();
            void reset();
            float getTime();
        private:
            int port0_len;
            int state_ = 1;
            uint16_t data_len_ = 0;
            uint16_t cur_len_ = 0;
            int id_ = 0;
            int type_ = 0;
            int port_ = 0;
            uint8_t check_sum_ = 0;
            int level_ = 0;
            // float overtime_ = 0.005;
            float overtime_;
            std::array<uint8_t, LEVEL_NUM> send_id_{0, 0, 0, 0};
            std::array<int, LEVEL_NUM> recv_flag_{1, 1, 1, 1};
            std::array<float, LEVEL_NUM> time_{0, 0, 0, 0};
            std::array<int, LEVEL_NUM> retimes_{0, 0, 0, 0};
            std::array<uint8_t, LEVEL_NUM> last_id_{0, 0, 0, 0};

            uint8_t level0_buff_[BUFF_SIZE / 4];
            uint8_t level1_buff_[BUFF_SIZE / 4];
            uint8_t level2_buff_[BUFF_SIZE / 4];
            uint8_t level3_buff_[BUFF_SIZE];

            uint8_t recv_buff_[BUFF_SIZE];
            uint8_t send_buff_[BUFF_SIZE];

            uint8_t spare_len_[LEVEL_NUM] = {BUFF_SIZE / 4, BUFF_SIZE / 4, BUFF_SIZE / 4, BUFF_SIZE};

            uint8_t *buff_[LEVEL_NUM];
            uint16_t max_len_[LEVEL_NUM] = {BUFF_SIZE / 4, BUFF_SIZE / 4, BUFF_SIZE / 4, BUFF_SIZE};

            port_func port_callback_[PORT_NUM];

            output_func output_ = nullptr;

            void type0Callback();
            void type1Callback();
            void type2Callback();
    };
}

#endif