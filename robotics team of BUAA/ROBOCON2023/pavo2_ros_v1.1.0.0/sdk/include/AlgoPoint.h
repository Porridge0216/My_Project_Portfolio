#ifndef ALGOPOINT_H
#define ALGOPOINT_H

#include <algorithm>
#include <cmath>
using namespace std;


class AlgoPoint
{
public:
    AlgoPoint() :m_x(0), m_y(0) {}
    AlgoPoint(float x, float y);
    AlgoPoint(const AlgoPoint& p);
    AlgoPoint& operator=(const AlgoPoint& p);
    float X() const;
    float Y() const;

private:
    float m_x, m_y;

};

#endif // ALGOPOINT_H
