//
// Created by 沈川江 on 2023/9/18.
//

#ifndef SC_REALTIME_KEYSTONE_SLIDINGWINDOW_H
#define SC_REALTIME_KEYSTONE_SLIDINGWINDOW_H

#include <vector>
//一共64个数据
#define MAX_SENSOR_NUM 64
//滑窗滤波的队列的数据个数
#define MAX_DATA_NUM 9
//每次滤波用的数据
#define WINDOW_DATA_NUM 5
//当两次滤波后数据小于error时，不更新数据
#define DataError 4  

int Filter_SlidingWindow(int index, int data);


#endif //SC_REALTIME_KEYSTONE_SLIDINGWINDOW_H
