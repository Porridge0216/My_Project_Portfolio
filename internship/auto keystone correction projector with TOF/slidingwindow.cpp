//
// Created by 沈川江 on 2023/9/18.
//

#include "slidingwindow.h"
#include <vector>
#include <../log.h>

extern int m_dataList[MAX_SENSOR_NUM][MAX_DATA_NUM] = {0};

void BubbleSort(int array[], int len) {
    int temp;
    for (int i = 0; i < len - 1; i++) {
        for (int j = 0; j < len - 1 - i; j++) {
            if (array[j] > array[j + 1]) {
                temp = array[j];
                array[j] = array[j + 1];
                array[j + 1] = temp;
            }
        }
    }
}

int Filter_SlidingWindow(int index, int data)
{
    static int dataNum[MAX_SENSOR_NUM] = {0};
    int i;
    int sum = 0;
    int out = 0;
    int array[MAX_DATA_NUM] = {0};

    for(i = MAX_DATA_NUM - 2; i >= 0; i--)
        m_dataList[index][i+1] = m_dataList[index][i];

    m_dataList[index][0] = data;
    if(dataNum[index] < MAX_DATA_NUM)
    {
        dataNum[index]++;
        for(i = 0; i < dataNum[index]; i++)
        {
            sum += m_dataList[index][i];
        }
        out = sum / dataNum[index];
    }
    else
    {
        for(i = 0; i < MAX_DATA_NUM; i++)
        {
            array[i] = m_dataList[index][i];
        }
        BubbleSort(array, MAX_DATA_NUM);

        int start = (MAX_DATA_NUM - WINDOW_DATA_NUM) / 2;

        for(i = start; i < start + WINDOW_DATA_NUM; i++)
        {
            sum += array[i];
        }
        out = sum / WINDOW_DATA_NUM;
    }
    return out;
}

/***************************************************************/

//当两次滤波后数据小于error时, 不更新数据, 已经放到了.h文件中
#define DataError 4  

vector<int> FilteredData_Last(64, 0);

void algorithm_realtime(vector<int> listdata){

    FilteredData.clear();
    for (int i = 0; i < listdata.size(); ++i){
        int res = Filter_SlidingWindow(i, listdata[i]);
        if(res - FilteredData_Last[i] < DataError){
            res = FilteredData_Last[i];
        }
        FilteredData_Last[i] = res;
        FilteredData.push_back(res);
    }
    ...
}
