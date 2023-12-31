

#ifndef _UNCERTAIN_TRANSFORMER_
#define _UNCERTAIN_TRANSFORMER_

#include "tf/transform_listener.h"
#include "ekf_pose_fusion/CovarianceTimeCache.h"
#include "ekf_pose_fusion/CovarianceStamped.h"

// ugly to inherit from listener but avoids multiple inheritance of Transformer in UncertainTransformListener

namespace uncertain_tf
{

    class UncertainTransformer : public tf::TransformListener
    {

    public:
        bool setCovariance(const StampedCovariance &cov);

        CovarianceTimeCache *getCovariance(unsigned int frame_number);

        std::vector<CovarianceTimeCache *> covariances_;

        mutable boost::recursive_mutex cov_mutex_;

        CompactFrameID lookupOrInsertFrameNumber(const std::string &frame_str);
    };

    static inline void covarianceStampedTFToMsg(const StampedCovariance &covariance, ekf_pose_fusion::CovarianceStamped &msg)
    {
        msg.header.stamp = covariance.stamp_;
        msg.header.frame_id = covariance.frame_id_;
        for (int row = 0; row < 6; ++row)
            for (int col = 0; col < 6; ++col)
                msg.covariance[row + col * 6] = covariance(row, col);
    };

    static inline void covarianceStampedMsgToTF(const ekf_pose_fusion::CovarianceStamped &msg, StampedCovariance &covariance)
    {
        covariance.stamp_ = msg.header.stamp;
        covariance.frame_id_ = msg.header.frame_id;
        // make sure matrix is sized well
        covariance.resize(6, 6);
        for (int row = 0; row < 6; ++row)
            for (int col = 0; col < 6; ++col)
                covariance(row, col) = msg.covariance[row + col * 6];
    };

} // namespace uncertain_tf

#endif
