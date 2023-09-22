
#ifndef _UNCERTAIN_TRANSFORM_BROADCASTER_H_
#define _UNCERTAIN_TRANSFORM_BROADCASTER_H_

#include "ekf_pose_fusion/UncertainTransformer.h"
#include "tf/transform_broadcaster.h"

namespace uncertain_tf
{

    class UncertainTransformBroadcaster : public tf::TransformBroadcaster
    {
    public:
        UncertainTransformBroadcaster();

        void sendCovariance(const StampedCovariance &covariance);
        void sendCovariance(const std::vector<StampedCovariance> &covariances);
        void sendCovariance(const std::vector<ekf_pose_fusion::CovarianceStamped> &covariances_msg);

    private:
        ros::NodeHandle node_;
        ros::Publisher publisher_;
    };

} // namespace tf

#endif
