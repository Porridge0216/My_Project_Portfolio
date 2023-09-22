#include "AlgoLine.h"
#include <cmath>

AlgoLine::AlgoLine(float a, float b, float c)
{
    m_a = a;
    m_b = b;
    m_c = c;
}

AlgoLine::AlgoLine(const AlgoLine& line)
{
    m_a = line.m_a;
    m_b = line.m_b;
    m_c = line.m_c;
}

AlgoLine& AlgoLine::operator=(const AlgoLine& line)
{
    m_a = line.m_a;
    m_b = line.m_b;
    m_c = line.m_c;

    return *this;
}

float AlgoLine::GetA()const
{
    return m_a;
}

float AlgoLine::GetB()const
{
    return m_b;
}

float AlgoLine::GetC()const
{
    return m_c;
}

float AlgoLine::CalculateDistance(AlgoPoint p)
{
    return static_cast<float>(abs(m_a * p.X() + m_b * p.Y() + m_c) / sqrt(pow(p.X(), 2)+pow(p.Y(), 2)));
}

bool AlgoLine::IsPointInline(AlgoPoint p)
{
    if(m_a*p.X()+m_b*p.Y()+m_c<static_cast<float>(1e-1))
    {
        return true;
    }
    return false;
}




