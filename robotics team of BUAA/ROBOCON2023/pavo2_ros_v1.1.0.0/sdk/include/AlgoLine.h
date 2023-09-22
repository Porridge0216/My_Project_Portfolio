#ifndef ALGOLINE_H
#define ALGOLINE_H
#include "AlgoPoint.h"
#include <vector>

#include <algorithm>
#include <cmath>
using namespace std;

class AlgoLine
{
public:
    AlgoLine();
    AlgoLine(float a,float b,float c);
    AlgoLine(const AlgoLine& line);
    AlgoLine& operator=(const AlgoLine& line);
    float GetA()const;
    float GetB()const;
    float GetC()const;
    float CalculateDistance(AlgoPoint p);
    bool IsPointInline(AlgoPoint p);

private:
    float m_a, m_b, m_c;

};



#endif // ALGOLINE_H
