#include "LdrAlgorithm.h"
//#include "PavoType.h"

LdrAlgorithm::LdrAlgorithm()
{
    InitTables();
}

/*
void LdrAlgorithm::HighRelectivityPreProcess(QVector<PointType>& points)
{

    int index_left = 0;
    int index_right = 0;
    int distance_max_index = 0;
    int distance_min_index = 0;
    int intensity_max_points_num = 0;

    bool inCalc = false;

    for(int i = 0; i < points.size(); i++) //first san preprocess
    {

        int threshold_index = points[i].distance / 1000; //unit:mm
        if(threshold_index > 20)
            threshold_index = 20;

        if(points[i].intensity >= intensity_threshold_table_[threshold_index])
        {
            if(!inCalc)
            {
                inCalc = true;
                index_left = i;
                index_right = i;
                distance_max_index = i;
                distance_min_index = i;
                intensity_max_points_num = 1;
            }
            else
            {
                index_right = i;
                if(points[i].distance > points[distance_max_index].distance)
                    distance_max_index = i;
                if(points[i].distance < points[distance_min_index].distance)
                    distance_min_index = i;
                if(points[i].intensity == INTENSITY_MAX)
                    intensity_max_points_num += 1;
            }
        }
        else //go out of intensity-special area
        {
            if(!inCalc)
                continue;

            inCalc = false;
            int index_mid = index_left + (index_right - index_left) / 2;
            if(index_right - index_left < 3 //points num < 3
               || intensity_max_points_num >= 5 //points number with I=2000 >= 5
               //|| (intensity_max_points_num == 0 && points[index_mid].distance < 1000) //distance < 100 && without I=2000 is 0
               || points[distance_max_index].distance - points[distance_min_index].distance > 300) //difference > 30cm
            {
                continue; //not satisfy the conditions
            }

            float angle_res = ANGLE_RESOLUTION * 3.1415926f/180;
            int n_left = 1;

            while(index_mid - n_left >= index_left)
            {
                if(points[index_mid - n_left].distance * n_left * angle_res > 30) //30 is 3cm
                    break;
                n_left += 1;
            }

            int n_right = 1;
            while(index_mid + n_right <= index_right)
            {
                if(points[index_mid + n_right].distance * n_right * angle_res > 30) //30 is 3cm
                    break;
                n_right += 1;
            }

            std::vector<AlgoPoint> suspicious_points;
            if(index_mid - n_left < index_left) //all left points in
            {
                for(int i = index_left; i <= index_mid; i++)
                {
                    suspicious_points.push_back(AlgoPoint(points[i].x, points[i].y));
                }
            }
            else
            {
                suspicious_points.push_back(AlgoPoint(points[index_mid - n_left].x, points[index_mid - n_left].y) );
                suspicious_points.push_back(AlgoPoint(points[index_mid].x, points[index_mid].y));
            }

            if(index_left + n_right > index_right)
            {
                for(int i = index_mid + 1; i <= index_right; i++)
                {
                    suspicious_points.push_back(AlgoPoint(points[i].x, points[i].y));
                }
            }
            else
            {
                suspicious_points.push_back(AlgoPoint(points[index_mid + n_right].x, points[index_mid + n_right].y));
            }


            //calculate line and do judge
            AlgoLine line_high_intensity = FitLine(suspicious_points);

            std::vector<AlgoPoint> fix_points;
            fix_points.push_back(AlgoPoint(points[index_mid].x, points[index_mid].y));
            fix_points.push_back(AlgoPoint(0, 0));

            AlgoLine line_pend = FitLine(fix_points);

            float angle_cos = AngleCos(line_high_intensity, line_pend);
            if(angle_cos >= 0.5) //0.5: 60°
            {
                for(i = index_left; i <= index_right; i++)
                {
                    points[i].intensity = INTENSITY_MAX;
                }
            }
        }
    }
}
*/

void LdrAlgorithm::HighRelectivityPreProcess(pavo_response_scan_t* points, int size)
{
    int index_left = 0;
    int index_right = 0;
    int distance_max_index = 0;
    int distance_min_index = 0;
    int intensity_max_points_num = 0;

    bool inCalc = false;

    std::vector<pavo_response_scan_t*> dat_ptr;
    for(int i = 0; i < size; i++)
    {
        dat_ptr.push_back(&points[i]);
    }

    for(int i = 0; i < size; i++) //first san preprocess
    {
        int threshold_index = points[i].distance / 500; //unit:2 mm
        if(threshold_index > 20)
            threshold_index = 20;

        if(points[i].intensity >= intensity_threshold_table_[threshold_index])
        {
            if(!inCalc)
            {
                inCalc = true;
                index_left = i;
                index_right = i;
                distance_max_index = i;
                distance_min_index = i;
                intensity_max_points_num = 1;
            }
            else
            {
                index_right = i;
                if(points[i].distance > points[distance_max_index].distance)
                    distance_max_index = i;
                if(points[i].distance < points[distance_min_index].distance)
                    distance_min_index = i;
                if(points[i].intensity == INTENSITY_MAX)
                    intensity_max_points_num += 1;
            }
        }
        else //go out of intensity-special area
        {
            if(!inCalc)
                continue;

            inCalc = false;
            int index_mid = index_left + (index_right - index_left) / 2;
            int points_num = index_right - index_left + 1;
            int distance_diff = points[distance_max_index].distance - points[distance_min_index].distance;
            if(index_right - index_left + 1 < 2 //points num < 3
               || intensity_max_points_num >= 5 //points number with I=2000 >= 5
               //|| (intensity_max_points_num == 0 && points[index_mid].distance < 1000) //distance < 100 && without I=2000 is 0
               || points[distance_max_index].distance - points[distance_min_index].distance > 300) //difference > 30cm
            {
                continue; //not satisfy the conditions
            }
#if 0
            float angle_res = ANGLE_RESOLUTION * 3.1415926f/360;
            int n_left = 1;

            while(index_mid - n_left >= index_left)
            {
                if(points[index_mid - n_left].distance * n_left * angle_res > 30) //30 is 3cm
                    break;
                n_left += 1;
            }

            int n_right = 1;
            while(index_mid + n_right <= index_right)
            {
                if(points[index_mid + n_right].distance * n_right * angle_res > 30) //30 is 3cm
                    break;
                n_right += 1;
            }

            float x,y;
            std::vector<AlgoPoint> suspicious_points;
            if(index_mid - n_left < index_left) //all left points in
            {
                for(int j = index_left; j <= index_mid; j++)
                {
                    x = points[j].distance * cos_lookup_table_[points[j].angle];
                    y = points[j].distance * sin_lookup_table_[points[j].angle];
                    suspicious_points.push_back(AlgoPoint(x, y));
                }
            }
            else
            {
                float x,y;
                x = points[index_mid - n_left].distance * cos_lookup_table_[points[index_mid - n_left].angle];
                y = points[index_mid - n_left].distance * sin_lookup_table_[points[index_mid - n_left].angle];
                suspicious_points.push_back(AlgoPoint(x, y));

                x = points[index_mid].distance * cos_lookup_table_[points[index_mid].angle];
                y = points[index_mid].distance * sin_lookup_table_[points[index_mid].angle];
                suspicious_points.push_back(AlgoPoint(x, y));
            }

            if(index_left + n_right > index_right)
            {
                for(int j = index_mid + 1; j <= index_right; j++)
                {
                    x = points[j].distance * cos_lookup_table_[points[j].angle];
                    y = points[j].distance * sin_lookup_table_[points[j].angle];
                    suspicious_points.push_back(AlgoPoint(x, y));
                }
            }
            else
            {
                x = points[index_mid + n_right].distance * cos_lookup_table_[points[index_mid + n_right].angle];
                y = points[index_mid + n_right].distance * sin_lookup_table_[points[index_mid + n_right].angle];
                suspicious_points.push_back(AlgoPoint(x, y));
            }


            //calculate line and do judge
            AlgoLine line_high_intensity = LdrAlgorithm::FitLine(suspicious_points);

            std::vector<AlgoPoint> fix_points;
            x = points[index_mid].distance * cos_lookup_table_[points[index_mid].angle];
            y = points[index_mid].distance * cos_lookup_table_[points[index_mid].angle];
            fix_points.push_back(AlgoPoint(x, y));
            fix_points.push_back(AlgoPoint(0, 0));

            AlgoLine line_pend = LdrAlgorithm::FitLine(fix_points);

            //float angle_cos = LdrAlgorithm::AngleCos(line_high_intensity, line_pend);
            //float angle_cos = LdrAlgorithm::AngleCos(line_high_intensity.GetA(), line_high_intensity.GetB(),
            //                                         line_pend.GetA(), line_pend.GetB());

            float A1 = line_high_intensity.GetA();
            float B1 = line_high_intensity.GetB();

            float A2 = line_pend.GetA();
            float B2 = line_pend.GetB();
            float angle_cos;

            float nominator = A1 * A2 + B1 * B2;

            float denominator = static_cast<float>(sqrt(pow(A1, 2) + pow(B1, 2))) *
                    static_cast<float>(sqrt(pow(A2, 2) + pow(B2, 2)));

            if(denominator < 0.0001) //means denominator == 0
                angle_cos = 1;
            else
                angle_cos = nominator / denominator;

            if(angle_cos > 0.5 || angle_cos < -0.5)
                continue;
#endif
            for(int j = index_left; j <= index_right; j++)
            {
                points[j].intensity = INTENSITY_MAX;
            }

        }

    }
}

void LdrAlgorithm::HighRelectivityFilter(pavo_response_scan_t* points, int size)
{
    HighRelectivityPreProcess(points, size);

#if 1
    int index_left = 0;
    int index_right = 0;
    int distance_max = 0;
    int distance_min = 0;
    int intensity_max = 0;
    int intensity_min = 0;
    int window_num = 0;

    bool inCalc = false;
    for(int i = 0; i < size; i++)
    {
        if(points[i].intensity == INTENSITY_MAX)
        {
            if(!inCalc)
            {
                inCalc = true;
                index_left = i;
            }

            index_right = i;
        }
        else
        {
            if(!inCalc)
                continue;

            inCalc = false;

            if(index_right - index_left + 1 < 3)//at least 3 points with I==2000
                continue;

            //process left side
            if(index_left >= 5)
            {
                window_num = 1;
                while(window_num < 50) //
                {
                    for(int w = 0; w <= FILTER_WINDOW_SIZE; w++)
                    {
                        int index = index_left - window_num - w;
                        if(points[index].distance > distance_max)
                            distance_max = points[index].distance;
                        if(points[index].distance < distance_min)
                            distance_min = points[index].distance;
                        if(points[index].intensity > intensity_max)
                            intensity_max = points[index].intensity;
                        if(points[index].intensity < intensity_min)
                            intensity_min = points[index].intensity;
                    }

                    if(distance_max - distance_min < 300 && intensity_max - intensity_min < 20) //found 30cm
                        break;

                    window_num++;
                }

                float distance_left = abs( (points[index_left - window_num].distance - points[index_left].distance) *
                        (points[index_left - window_num].angle - points[index_left].angle) * 3.1415926 / 180.0f );

                if(window_num < 50 && distance_left > 300) //30cm
                {
                    for(int j = index_left - window_num; j < index_left; j++)
                    {
                        points[j].distance = 0;
                        points[j].intensity = 0;
                    }
                }
            }

            // process right side
            if(index_right + 5 < size)
            {
                window_num = 1;
                while(window_num < 50) //
                {
                    for(int w = 0; w <= FILTER_WINDOW_SIZE; w++)
                    {
                        int index = index_right + window_num + w;
                        if(points[index].distance > distance_max)
                            distance_max = points[index].distance;
                        if(points[index].distance < distance_min)
                            distance_min = points[index].distance;
                        if(points[index].intensity > intensity_max)
                            intensity_max = points[index].intensity;
                        if(points[index].intensity < intensity_min)
                            intensity_min = points[index].intensity;
                    }

                    if(distance_max - distance_min < 300 && intensity_max - intensity_min < 20) //found
                        break;

                    window_num++;
                }

                float distance_right = abs( (points[index_right + window_num].distance - points[index_right].distance) *
                        (points[index_right + window_num].angle - points[index_right].angle) * 3.1415926 / 180.0f);

                if(window_num < 50 && distance_right > 300 ) //30cm
                {
                    for(int j = index_right ; j < index_right + window_num; j++)
                    {
                        points[j].distance = 0;
                        points[j].intensity = 0;
                    }
                }
            }

        }
    }
#endif
}

/*
void LdrAlgorithm::HighRelectivityFilter(QVector<PointType>& points)
{
    HighRelectivityPreProcess(points);
#if 0
    int index_left = 0;
    int index_right = 0;
    int distance_max = 0;
    int distance_min = 0;
    int intensity_max = 0;
    int intensity_min = 0;
    int window_num = 0;

    bool inCalc = false;
    for(int i = 0; i < points.size(); i++)
    {
        if(points[i].intensity == INTENSITY_MAX)
        {
            if(!inCalc)
            {
                inCalc = true;
                index_left = i;
            }

            index_right = i;
        }
        else
        {
            if(!inCalc)
                continue;

            inCalc = false;
            window_num = 0;
            if(index_right - index_left < 3)//at least 3 points with I==2000
                continue;

            //need to deal with 360° case
            if(index_left < 5 || index_right + 5 > points.size())//will exceed the boundary of the points
                continue;

            //process left side
            window_num = 1;
            while(window_num < 50) //
            {
                for(int w = 0; w <= FILTER_WINDOW_SIZE; w++)
                {
                    int index = index_left - window_num - w;
                    if(points[index].distance > distance_max)
                        distance_max = points[index].distance;
                    if(points[index].distance < distance_min)
                        distance_min = points[index].distance;
                    if(points[index].intensity > intensity_max)
                        intensity_max = points[index].intensity;
                    if(points[index].intensity < intensity_min)
                        intensity_min = points[index].intensity;
                }

                if(distance_max - distance_min < 300 && intensity_max - intensity_min < 20) //found 30cm
                    break;

                window_num++;
            }

            if(points[index_left - window_num].distance - points[index_left].distance > 300 ) //30cm
            {
                for(int j = index_left - window_num; j < index_left; j++)
                {
                    points[j].distance = 0;
                    points[j].intensity = 0;
                    points[j].x = points[j].y = points[j].z = 0;
                }
            }

            // process right side
            window_num = 1;
            while(window_num < 50) //
            {
                for(int w = 0; w <= FILTER_WINDOW_SIZE; w++)
                {
                    int index = index_right + window_num + w;
                    if(points[index].distance > distance_max)
                        distance_max = points[index].distance;
                    if(points[index].distance < distance_min)
                        distance_min = points[index].distance;
                    if(points[index].intensity > intensity_max)
                        intensity_max = points[index].intensity;
                    if(points[index].intensity < intensity_min)
                        intensity_min = points[index].intensity;
                }

                if(distance_max - distance_min < 300 && intensity_max - intensity_min < 20) //found
                    break;

                window_num++;
            }

            if(points[index_right + window_num].distance - points[index_right].distance > 300 ) //30cm
            {
                for(int j = index_right ; j < index_right + window_num; j++)
                {
                    points[j].distance = 0;
                    points[j].intensity = 0;
                    points[j].x = points[j].y = points[j].z = 0;
                }
            }

        }
    }
#endif
}
*/

#define HDL_Grabber_toRadians(x) ((x) * 3.1415926 / 180.0)
const int NUM_ROT_ANGLES = 36001;
void LdrAlgorithm::InitTables()
{
    int radius = 0;
#if 0
    if (intensity_threshold_table_.size() == 0)
    {
        intensity_threshold_table_.resize(INTENSITY_THRESHOLD_TABLE_LENGTH + 1);
        for(int i = 0; i <= INTENSITY_THRESHOLD_TABLE_LENGTH; i++)
        {
            if(i < 5)
                radius = 5;
            else if(i > 20)
                radius = 20;
            else
                radius = i;
            intensity_threshold_table_[i] = -10 * radius + 650;
        }
    }
#endif
    if (intensity_threshold_table_.size() == 0)
    {
        intensity_threshold_table_.resize(INTENSITY_THRESHOLD_TABLE_LENGTH + 1);
        intensity_threshold_table_[0] = 600;
        intensity_threshold_table_[1] = 590;
        intensity_threshold_table_[2] = 590;
        intensity_threshold_table_[3] = 560;
        intensity_threshold_table_[4] = 560;
        intensity_threshold_table_[5] = 550;
        intensity_threshold_table_[6] = 550;
        intensity_threshold_table_[7] = 520;
        intensity_threshold_table_[8] = 520;
        intensity_threshold_table_[9] = 510;
        intensity_threshold_table_[10] = 510;
        intensity_threshold_table_[11] = 510;
        intensity_threshold_table_[12] = 510;
        intensity_threshold_table_[13] = 500;
        intensity_threshold_table_[14] = 500;
        intensity_threshold_table_[15] = 490;
        intensity_threshold_table_[16] = 490;
        intensity_threshold_table_[17] = 490;
        intensity_threshold_table_[18] = 490;
        intensity_threshold_table_[19] = 470;
        intensity_threshold_table_[20] = 470;
        intensity_threshold_table_[21] = 450;
        intensity_threshold_table_[22] = 450;
        intensity_threshold_table_[23] = 440;
        intensity_threshold_table_[24] = 440;
    }

    if (cos_lookup_table_.size() == 0 || sin_lookup_table_.size() == 0)
    {
      cos_lookup_table_.resize(NUM_ROT_ANGLES);
      sin_lookup_table_.resize(NUM_ROT_ANGLES);


      for (unsigned int i = 0; i < NUM_ROT_ANGLES; i++)
      {
        double rad = HDL_Grabber_toRadians(i / 100.0);
        cos_lookup_table_[i] = std::cos(rad);
        sin_lookup_table_[i] = std::sin(rad);
      }
    }
}


AlgoLine LdrAlgorithm::FitLine(std::vector<AlgoPoint>& points)
{
    float sum_x = 0, sum_y = 0, A = 0, B = 0, beta = 0;
    for(unsigned int i=0; i<points.size(); i++)
    {
        sum_x += points[i].X();
        sum_y += points[i].Y();
    }
    sum_x = sum_x / points.size();
    sum_y = sum_y / points.size();

    for (unsigned int i = 0; i < points.size(); i++)
    {
        A += pow(sum_x - points[i].X(), 2);
        B += (sum_y - points[i].Y())*(sum_x - points[i].X());
    }
    beta = static_cast<float>(sqrt(pow(A, 2) + pow(B, 2)));
    const float a = static_cast<float>(-B / beta);
    const float b = static_cast<float>(A / beta);
    const float c = -a * sum_x - b * sum_y;
    AlgoLine line(a, b, c);
    return line;
}


float LdrAlgorithm::AngleCos(const AlgoLine& line1, const AlgoLine& line2)
{
    float A1 = line1.GetA();
    float B1 = line1.GetB();

    float A2 = line2.GetA();
    float B2 = line2.GetB();

    float nominator = A1 * A2 + B1 * B2;

    float denominator = static_cast<float>(sqrt(pow(A1, 2) + pow(B1, 2))) *
            static_cast<float>(sqrt(pow(A2, 2) + pow(B2, 2)));

    if(denominator < 0.0001) //means denominator == 0
        return 1.0f;

    return nominator / denominator;
}

float LdrAlgorithm::AngleCos(float A1, float B1, float A2, float B2)
{

    float nominator = A1 * A2 + B1 * B2;

    float denominator = static_cast<float>(sqrt(pow(A1, 2) + pow(B1, 2))) *
            static_cast<float>(sqrt(pow(A2, 2) + pow(B2, 2)));

    if(denominator < 0.0001) //means denominator == 0
        return 1.0f;

    return nominator / denominator;
}

