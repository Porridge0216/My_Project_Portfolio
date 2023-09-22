
#include "ekf_pose_fusion/UncertainTransformBroadcaster.h"
#include "ekf_pose_fusion/utfMessage.h"

using namespace Eigen;
using namespace tf;
using namespace uncertain_tf;
using namespace ekf_pose_fusion;
using namespace std;

namespace uncertain_tf
{

    UncertainTransformBroadcaster::UncertainTransformBroadcaster()
    {
        publisher_ = node_.advertise<ekf_pose_fusion::utfMessage>("/tf_uncertainty", 100);
        ros::NodeHandle l_nh("~");
        // tf_prefix_ = getPrefixParam(l_nh);
    }

    void UncertainTransformBroadcaster::sendCovariance(const StampedCovariance &covariance)
    {
        std::vector<StampedCovariance> v1;
        v1.push_back(covariance);
        sendCovariance(v1);
    }

    void UncertainTransformBroadcaster::sendCovariance(const std::vector<StampedCovariance> &covariances)
    {
        std::vector<ekf_pose_fusion::CovarianceStamped> utfm;
        for (std::vector<StampedCovariance>::const_iterator it = covariances.begin(); it != covariances.end(); ++it)
        {
            ekf_pose_fusion::CovarianceStamped msg;
            covarianceStampedTFToMsg(*it, msg);
            utfm.push_back(msg);
        }
        sendCovariance(utfm);
    }

    void UncertainTransformBroadcaster::sendCovariance(const std::vector<ekf_pose_fusion::CovarianceStamped> &covariances_msg)
    {
        ekf_pose_fusion::utfMessage msg;
        msg.covariances = covariances_msg;
        // std::cout << "publishing:" << endl << msg << endl;
        publisher_.publish(msg);
    }

}
