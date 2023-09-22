#ifndef LDRALGORITHM_H
#define LDRALGORITHM_H
//#include <QVector>
//#include <PavoType.h>
#include <vector>
#include "pavo2_types.h"
#include "AlgoLine.h"
#include <algorithm>
#include <cmath>
using namespace std;


const int INTENSITY_THRESHOLD_TABLE_LENGTH = 24;
const int INTENSITY_MAX = 2000;
const int FILTER_WINDOW_SIZE = 5;

class LdrAlgorithm
{
public:
    LdrAlgorithm();
    void InitTables();
    //void HighRelectivityPreProcess(QVector<PointType>& points);
    //void HighRelectivityFilter(QVector<PointType>& points);

    void HighRelectivityPreProcess(pavo_response_scan_t* points, int size);
    void HighRelectivityFilter(pavo_response_scan_t* points, int size);

    static AlgoLine FitLine(std::vector<AlgoPoint>& points);

    static float AngleCos(const AlgoLine& line1, const AlgoLine& line2);

    static float AngleCos(float A1, float B1, float A2, float B2);

private:

private:
    std::vector<float> intensity_threshold_table_;

    std::vector<double> cos_lookup_table_;
    std::vector<double> sin_lookup_table_;
};

#endif // LDRALGORITHM_H
