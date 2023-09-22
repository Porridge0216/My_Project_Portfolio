#include "AlgoPoint.h"

AlgoPoint::AlgoPoint(float x,float y)
{
    this->m_x = x;
    this->m_y = y;
}
AlgoPoint::AlgoPoint(const AlgoPoint &p)
{
    m_x = p.m_x;
    m_y = p.m_y;
}
AlgoPoint& AlgoPoint::operator=(const AlgoPoint& p)
{
    m_x = p.m_x;
    m_y = p.m_y;

    return *this;
}
float AlgoPoint::X() const
{
    return m_x;
}

float AlgoPoint::Y() const
{
    return m_y;
}

