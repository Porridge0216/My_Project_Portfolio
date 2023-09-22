

#include "ekf_pose_fusion/CovarianceTimeCache.h"

using namespace Eigen;
using namespace tf;
using namespace uncertain_tf;
using namespace std;

CovarianceStorage::CovarianceStorage(const MatrixXd &data, ros::Time stamp, CompactFrameID frame_id, CompactFrameID child_frame_id)
    : covariance_(data), stamp_(stamp), frame_id_(frame_id), child_frame_id_(child_frame_id_)
{
}

CovarianceTimeCache::CovarianceTimeCache(ros::Duration max_storage_time)
    : max_storage_time_(max_storage_time)
{
}

bool CovarianceTimeCache::getData(ros::Time time, CovarianceStorage &data_out, std::string *error_str)
{
    const CovarianceStorage *p_temp_1 = NULL;
    const CovarianceStorage *p_temp_2 = NULL;

    int num_nodes = findClosest(p_temp_1, p_temp_2, time, error_str);
    if (num_nodes == 0)
    {
        return false;
    }
    else if (num_nodes == 1)
    {
        data_out = *p_temp_1;
    }
    else if (num_nodes == 2)
    {
        // if( p_temp_1->frame_id_ == p_temp_2->frame_id_)
        //{
        // TODO 协方差的插值
        interpolate(*p_temp_1, *p_temp_2, time, data_out);
        //}
        // else
        //{
        //    data_out = *p_temp_1;
        //}
    }
    else
    {
#ifdef ROS_BREAK
        ROS_BREAK();
#endif
    }

    return true;
}

bool CovarianceTimeCache::insertData(const CovarianceStorage &new_data)
{
    // std::cout << "insertData new_data.stamp_ " << new_data.stamp_ << endl;
    auto storage_it = storage_.begin();

    if (storage_it != storage_.end())
    {
        if (storage_it->stamp_ > new_data.stamp_ + max_storage_time_)
            return false;
    }

    // 从begin->end，时间依次逐渐变小，即时间最新的在begin
    while (storage_it != storage_.end())
    {
        if (storage_it->stamp_ <= new_data.stamp_)
            break;
        storage_it++;
    }

    // 找到一帧和输入data时间戳相同的，报告重复
    if (storage_it != storage_.end() && storage_it->stamp_ == new_data.stamp_)
    {

        return false;
    }
    else
    {
        storage_.insert(storage_it, new_data);
    }

    std::cout << "storage size " << storage_.size() << endl;

    pruneList();
    return true;
}

void CovarianceTimeCache::clearList()
{
    storage_.clear();
}

uint8_t CovarianceTimeCache::findClosest(const CovarianceStorage *&one, const CovarianceStorage *&two, ros::Time target_time, std::string *error_str)
{
    // No values stored
    if (storage_.empty())
    {
        // createEmptyException(error_str);
        return 0;
    }

    // If time == 0 return the latest
    if (target_time.isZero())
    {
        one = &storage_.front();
        return 1;
    }

    // One value stored
    if (++storage_.begin() == storage_.end())
    {
        // CovarianceStorage& ts = *storage_.begin();
        const CovarianceStorage &ts = *storage_.begin();
        if (ts.stamp_ == target_time)
        {
            one = &ts;
            return 1;
        }
        else
        {
            // createExtrapolationException1(target_time, ts.stamp_, error_str);
            return 0;
        }
    }

    ros::Time latest_time = (*storage_.begin()).stamp_;
    ros::Time earliest_time = (*(storage_.rbegin())).stamp_;

    if (target_time == latest_time)
    {
        one = &(*storage_.begin());
        return 1;
    }
    else if (target_time == earliest_time)
    {
        one = &(*storage_.rbegin());
        return 1;
    }
    // Catch cases that would require extrapolation
    else if (target_time > latest_time)
    {
        // createExtrapolationException2(target_time, latest_time, error_str);
        ROS_ERROR_STREAM("tar: " << target_time << "  latest: " << latest_time << endl);
        ROS_ERROR("EXTRAPOLATION TO FUTURE REQ");
        return 0;
    }
    else if (target_time < earliest_time)
    {
        // createExtrapolationException3(target_time, earliest_time, error_str);
        ROS_ERROR("EXTRAPOLATION TO PAST REQ");
        ROS_ERROR_STREAM(target_time << ", " << earliest_time);
        return 0;
    }

    // Create a temporary object to compare to when searching the lower bound via std::set
    CovarianceStorage tmp;
    tmp.stamp_ = target_time;

    // Find the first value equal or higher than the target value
    L_CovarianceStorage::iterator storage_it =
        std::lower_bound(
            storage_.begin(),
            storage_.end(),
            tmp, std::greater<CovarianceStorage>());

    // Finally the case were somewhere in the middle  Guarenteed no extrapolation :-)
    two = &*(storage_it);   // Older
    one = &*(--storage_it); // Newer

    return 2;
}

void CovarianceTimeCache::interpolate(const CovarianceStorage &one, const CovarianceStorage &two, ros::Time time, CovarianceStorage &output)
{
    output = two; // TODO
}

// 用于维护修剪时间序列，删除超过最大储存时间的内容
void CovarianceTimeCache::pruneList()
{
    ros::Time latest_time = storage_.begin()->stamp_;

    while (!storage_.empty() && storage_.back().stamp_ + max_storage_time_ < latest_time)
    {
        storage_.pop_back();
    }
}
