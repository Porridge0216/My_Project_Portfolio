
#ifndef __BR_POSE_EKF__
#define __BR_POSE_EKF__

// ros stuff
#include <ros/ros.h>
#include <tf/tf.h>
#include <tf/transform_listener.h>
#include <tf/transform_broadcaster.h>
#include "odom_estimation.h"

// messages
#include "nav_msgs/Odometry.h"
#include "geometry_msgs/Twist.h"
#include "sensor_msgs/Imu.h"
#include "geometry_msgs/PoseStamped.h"
#include "geometry_msgs/PoseWithCovarianceStamped.h"

#include <boost/thread/mutex.hpp>
#include "ekf_pose_fusion/CovarianceTimeCache.h"

// log files
#include <fstream>

#include <mutex>
#include <algorithm>

namespace estimation
{

    typedef boost::shared_ptr<nav_msgs::Odometry const> wheel_odomConstPtr;
    typedef boost::shared_ptr<geometry_msgs::PoseWithCovarianceStamped const> visual_odomConstPtr;
    typedef boost::shared_ptr<geometry_msgs::PoseWithCovarianceStamped const> laser_odomConstPtr;
    // utils functions
    static void angleOverflowCorrect(double &a, double ref)
    {
        // while ((a - ref) > M_PI)
        //     a -= 2 * M_PI;
        // while ((a - ref) < -M_PI)
        //     a += 2 * M_PI;

        while (fabs(a - ref) > fabs(a - 2 * M_PI - ref))
        {
            a -= 2 * M_PI;
        }
        while (fabs(a - ref) > fabs(a + 2 * M_PI - ref))
        {
            a += 2 * M_PI;
        }
    }
    static void customizAngle_in_fabsPi(double &an)
    {
        while (an > M_PI)
            an -= 2 * M_PI;
        while (an < -M_PI)
            an += 2 * M_PI;
    }
    static void ColumnVector2Transform(MatrixWrapper::ColumnVector &state, tf::Transform &trans)
    {
        trans.setOrigin(tf::Vector3(state(1), state(2), 0));
        tf::Quaternion q;
        q.setRPY(0, 0, state(3));
        q.normalize();
        trans.setRotation(q);
    }
    class pose_factor
    {
    public:
        ros::Time stamp;
        tf::Transform woTrans;
        Eigen::Matrix3d cov;
        tf::Transform measurement;
        using Ptr = std::shared_ptr<pose_factor>;
    };
    class pose_fuser;
    class BR_transformer
    {
        boost::mutex m2o_mutex;
        boost::mutex o2b_mutex;
        boost::mutex m2b_cov_mutex;
        bool fuserSeted = false;

    public:
        tf::StampedTransform map2odom;
        tf::StampedTransform odom2basefootprint;
        std::string map_frame;            // CompactFrameID as 1
        std::string odom_frame;           // CompactFrameID as 2
        std::string base_footprint_frame; // CompactFrameID as 3
        std::string laser_frame;          // CompactFrameID as 4
        std::string camera_frame;         // CompactFrameID as 5
        tf::TimeCache m2o_list;
        tf::TimeCache o2b_list;
        tf::TimeCache true_pose_list;
        uncertain_tf::CovarianceTimeCache m2b_cov_list;
        uncertain_tf::CovarianceTimeCache m2o_cov_list;
        class pose_fuser *fuser;

    public:
        void setFuser(class pose_fuser *ptr)
        {
            fuser = ptr;
            fuserSeted = true;
        }
        BR_transformer()
        {
            getFrameParams();
            map2odom.setIdentity();
            map2odom.frame_id_ = map_frame;
            map2odom.child_frame_id_ = odom_frame;
            odom2basefootprint.setIdentity();
            odom2basefootprint.frame_id_ = odom_frame;
            odom2basefootprint.child_frame_id_ = base_footprint_frame;
        }
        void getFrameParams()
        {
            ros::NodeHandle n_pri("~");
            n_pri.param<std::string>("map_frame", map_frame, "map");
            n_pri.param<std::string>("base_footprint_frame", base_footprint_frame, "base_footprint");
            n_pri.param<std::string>("laser_frame", laser_frame, "laser");
            n_pri.param<std::string>("camera_frame", camera_frame, "camera");
            n_pri.param<std::string>("odom_frame", odom_frame, "odom");
        }

        void set_m2o(double x, double y, double alpha, const ros::Time &stamp)
        {
            boost::mutex::scoped_lock(m2o_mutex);

            map2odom.setOrigin(tf::Vector3(x, y, 0));
            map2odom.setRotation(tf::createQuaternionFromRPY(0, 0, alpha));
            map2odom.stamp_ = stamp;
            m2o_list.insertData(tf::TransformStorage(map2odom, 1, 2));
        }

        void set_o2b(double x, double y, double alpha, const ros::Time &stamp)
        {
            boost::mutex::scoped_lock(o2b_mutex);

            odom2basefootprint.setOrigin(tf::Vector3(x, y, 0));
            tf::Quaternion q;
            q.setRPY(0, 0, alpha);
            q.normalize();
            odom2basefootprint.setRotation(q);
            odom2basefootprint.stamp_ = stamp;
            if (!o2b_list.insertData(tf::TransformStorage(odom2basefootprint, 2, 3)))
            {
                ROS_ERROR("set failed");
                std::cout << "lastest: " << o2b_list.getLatestTimestamp() << std::endl;
                std::cout << "this: " << stamp << std::endl;
            }
        }
        void decomposeTransform(const tf::Transform &trans, MatrixWrapper::ColumnVector &vec)
        {
            vec(1) = trans.getOrigin().x();
            vec(2) = trans.getOrigin().y();
            vec(3) = getYaw(trans.getRotation());
        };
        void compensate_m2o(const tf::Transform &comp, const ros::Time &stamp)
        {
            boost::mutex::scoped_lock(m2o_mutex);

            map2odom *= comp;
            map2odom.stamp_ = stamp;
            m2o_list.insertData(tf::TransformStorage(map2odom, 1, 2));
            MatrixWrapper::ColumnVector m2ovec(3);
            decomposeTransform(map2odom, m2ovec);
            std::cout << "compen m2ovec: " << m2ovec << std::endl;
        }
        void set_m2o(const tf::Transform &trans, const ros::Time &stamp)
        {
            boost::mutex::scoped_lock(m2o_mutex);

            map2odom.setRotation(trans.getRotation().normalize());
            map2odom.setOrigin(trans.getOrigin());
            map2odom.stamp_ = stamp;

            // 此处出现segmentation fault, 时间戳跳跃问题
            m2o_list.insertData(tf::TransformStorage(map2odom, 1, 2));
        }
        void set_o2b(const tf::Transform &trans, const ros::Time &stamp)
        {
            boost::mutex::scoped_lock(o2b_mutex);

            odom2basefootprint.setRotation(trans.getRotation().normalize());
            odom2basefootprint.setOrigin(trans.getOrigin());
            odom2basefootprint.stamp_ = stamp;
            if (!o2b_list.insertData(tf::TransformStorage(odom2basefootprint, 2, 3)))
            {
                ROS_ERROR("set failed");

                std::cout << "lastest: " << o2b_list.getLatestTimestamp() << std::endl;
                std::cout << "this: " << stamp << std::endl;
            }
        }

        void pub_m2o(const ros::Time &stamp)
        {
            boost::mutex::scoped_lock(m2o_mutex);

            map2odom.stamp_ = stamp;
        }

        void pub_o2b(const ros::Time &stamp)
        {
            boost::mutex::scoped_lock(o2b_mutex);

            odom2basefootprint.stamp_ = stamp;
        }

        void set_m2b_cov(const Eigen::MatrixXd &cov, const ros::Time &stamp)
        {
            using namespace uncertain_tf;
            boost::mutex::scoped_lock(o2b_cov_mutex);

            CovarianceStorage covsto(cov, stamp, 1, 3);
            m2b_cov_list.insertData(covsto);
        }

        void lookupPoseAndCov(const ros::Time &time, tf::Transform &transform, uncertain_tf::CovarianceStorage &cs)
        {

            // TODO 记得改这里的插值
            tf::TransformStorage m2o, o2b;

            {
                boost::mutex::scoped_lock(m2o_mutex);
                boost::mutex::scoped_lock(o2b_mutex);

                if ((!m2o_list.getData(time, m2o)) || (!o2b_list.getData(time, o2b)))
                {
                    m2o_list.getData(ros::Time(0), m2o);
                    o2b_list.getData(ros::Time(0), o2b);
                    ROS_ERROR("it is trans");
                }

                transform = map2odom * tf::Transform(o2b.rotation_, o2b.translation_);
            }
            {
                boost::mutex::scoped_lock(m2b_cov_mutex);

                if (!m2b_cov_list.getData(time, cs))
                {
                    m2b_cov_list.getData(ros::Time(0), cs);
                    ROS_ERROR("it is cov");
                }
            }

            // listener.getCovariance(listener.lookupOrInsertFrameNumber(base_footprint_frame))->getData(time, cs);
        }
    };
    class pose_fuser
    {
    public:
        bool transformerSeted = false;
        int fuse_list_length = 10;
        ros::Time last_filt_time;
        tf::Transform last_filt_woTrans;

        std::vector<pose_factor::Ptr> factor_list;
        class BR_transformer *trans_ptr;
        BFL::LinearAnalyticSystemModelGaussianUncertainty *sys_model_;
        BFL::LinearAnalyticConditionalGaussian *sys_pdf_;
        BFL::LinearAnalyticConditionalGaussian *meas_pdf_;
        BFL::LinearAnalyticMeasurementModelGaussianUncertainty *meas_model_;
        BFL::Gaussian *prior_;
        BFL::ExtendedKalmanFilter *filter_;

    public:
        pose_fuser()
        {
            using namespace MatrixWrapper;
            using namespace BFL;
            ColumnVector sysNoise_Mu(3);
            sysNoise_Mu = 0;
            SymmetricMatrix sysNoise_Cov(3);
            // sys的过程噪声设置为0，实际的过程噪声直接在updateCov中加入，故过程噪声就不再重复叠加
            sysNoise_Cov = 0;

            Matrix A(3, 3);
            A = 0;
            A(1, 1) = 1.0;
            A(2, 2) = 1.0;
            A(3, 3) = 1.0;
            Matrix B(3, 3);
            B = 0;
            B(1, 1) = 1.0;
            B(2, 1) = 1.0;
            B(3, 3) = 1.0;

            std::vector<Matrix> AB(2);
            AB[0] = A;
            AB[1] = B;

            Gaussian system_Uncertainty(sysNoise_Mu, sysNoise_Cov);
            sys_pdf_ = new LinearAnalyticConditionalGaussian(AB, system_Uncertainty);
            sys_model_ = new LinearAnalyticSystemModelGaussianUncertainty(sys_pdf_);

            ColumnVector measNoise_Mu(3);
            measNoise_Mu = 0;
            SymmetricMatrix measNoise_Cov(3);
            measNoise_Cov = 0;
            for (unsigned int i = 1; i <= 3; i++)
                measNoise_Cov(i, i) = pow(0.008, 2);
            Gaussian measurement_Uncertainty(measNoise_Mu, measNoise_Cov);
            Matrix H(3, 3);
            H = 0;
            H(1, 1) = 1;
            H(2, 2) = 1;
            H(3, 3) = 1;
            meas_pdf_ = new LinearAnalyticConditionalGaussian(H, measurement_Uncertainty);
            meas_model_ = new LinearAnalyticMeasurementModelGaussianUncertainty(meas_pdf_);
        }
        void setTransformer(class BR_transformer *_ptr)
        {
            trans_ptr = _ptr;
            transformerSeted = true;
        }
        void initFilter(const tf::Transform &prior, const ros::Time &time)
        {
            using namespace MatrixWrapper;
            using namespace BFL;
            ColumnVector prior_Mu(3);
            decomposeTransform(prior, prior_Mu);
            SymmetricMatrix prior_Cov(3);
            for (unsigned int i = 1; i <= 3; i++)
            {
                for (unsigned int j = 1; j <= 3; j++)
                {
                    if (i == j)
                        prior_Cov(i, j) = pow(0.01, 2);
                    else
                        prior_Cov(i, j) = 0;
                }
            }
            prior_ = new Gaussian(prior_Mu, prior_Cov);
            filter_ = new ExtendedKalmanFilter(prior_);
        }
        void updateCov(ros::Time &last, ros::Time &now, tf::Transform &lastwo, tf::Transform &thiswo)
        // 直接对filter_进行处理
        {
            MatrixWrapper::SymmetricMatrix cov(3);
            cov = 0;
            for (int i = 1; i <= 3; ++i)
            {
                cov(i, i) = 1e-5;
            }
            filter_->PostGet()->CovarianceSet(cov);
        }
        void decomposeTransform(const tf::Transform &trans, MatrixWrapper::ColumnVector &vec)
        {
            vec(1) = trans.getOrigin().x();
            vec(2) = trans.getOrigin().y();
            vec(3) = getYaw(trans.getRotation());
        };
        MatrixWrapper::SymmetricMatrix eigen2wraped(const Eigen::MatrixXd &eig)
        {
            MatrixWrapper::SymmetricMatrix ret(eig.rows());
            for (unsigned int i = 0; i < eig.rows(); i++)
                for (unsigned int j = 0; j < eig.rows(); j++)
                    ret(i + 1, j + 1) = eig(i, j);
            return ret;
        }
        void ColumnVector2Transform(MatrixWrapper::ColumnVector &state, tf::Transform &trans)
        {
            trans.setOrigin(tf::Vector3(state(1), state(2), 0));
            tf::Quaternion q;
            q.setRPY(0, 0, state(3));
            q.normalize();
            trans.setRotation(q);
        }
        void fuseList(void)
        {
            using namespace std;
            // 对已经排序的list进行融合（其实排序不是必须的）
            // get current wo trans, 需要与transformer进行交互，应该进一步设计类的关系
            // 调用 getCurrentWoTF
            tf::Transform curr_m2o, curr_o2b;
            getCurrentTF(curr_o2b, curr_m2o, factor_list.back()->stamp);

            // 设置为自上次校准之后的wo叠加值
            MatrixWrapper::ColumnVector pripose(3);
            decomposeTransform(curr_m2o * curr_o2b, pripose);
            filter_->PostGet()->ExpectedValueSet(pripose);
            std::cout << pripose << std::endl;
            // 先验协方差的更新是与时间间隔、位移变化量有关的
            updateCov(last_filt_time, factor_list.back()->stamp, last_filt_woTrans, curr_o2b);
            cout << filter_->PostGet()->CovarianceGet() << endl;

            MatrixWrapper::ColumnVector meas(3);
            for (int i = 0; i < factor_list.size(); i++)
            {

                decomposeTransform(factor_list[i]->measurement * factor_list[i]->woTrans.inverse() * curr_o2b, meas);
                angleOverflowCorrect(meas(3), filter_->PostGet()->ExpectedValueGet()(3));
                meas_pdf_->AdditiveNoiseSigmaSet(eigen2wraped(factor_list[i]->cov));
                cout << meas << endl;
                filter_->Update(sys_model_, MatrixWrapper::ColumnVector(3) = 0, meas_model_, meas);
                if (i == (factor_list.size() - 1))
                {
                    customizAngle_in_fabsPi(filter_->PostGet()->ExpectedValueGet()(3));
                }
                cout << filter_->PostGet()->CovarianceGet() << endl;
                cout << filter_->PostGet()->ExpectedValueGet() << endl;
            }

            tf::Transform posttf;
            MatrixWrapper::ColumnVector postpose = filter_->PostGet()->ExpectedValueGet();
            cout << postpose << endl;
            cout << filter_->PostGet()->CovarianceGet() << endl;
            ColumnVector2Transform(postpose, posttf);

            // 同时用最新的时间戳对last_filt_time进行更新
            last_filt_time = factor_list.back()->stamp;
            last_filt_woTrans = factor_list.back()->woTrans;

            // 利用融合值更新transformer中的odom2base_footprint变换
            // 调用 setCurrentWoTF
            setCurrentM2oTF(posttf * curr_o2b.inverse(), last_filt_time);
        }

        void clearList(void)
        {
            // 使用了shared_ptr，不用对list中的所有factor都释放
            factor_list.clear();
        }

        //      早     ->    晚
        // list[begin] -> list[end]
        void sortList(void)
        {
            std::sort(factor_list.begin(), factor_list.end(),
                      [](const pose_factor::Ptr &a, const pose_factor::Ptr &b)
                      { return a->stamp < b->stamp; });
        }

        void getCurrentTF(tf::Transform &currentWoTf, tf::Transform &currentM2oTf, ros::Time &stamp)

        {
            assert(transformerSeted);
            tf::TransformStorage tr;
            // TODO wo tf stamp should be same with the input stamp
            if (!trans_ptr->o2b_list.getData(stamp, tr))
            {
                trans_ptr->o2b_list.getData(ros::Time(0), tr);
            }
            currentWoTf.setRotation(tr.rotation_);
            currentWoTf.setOrigin(tr.translation_);

            currentM2oTf = trans_ptr->map2odom;
        }

        void setCurrentM2oTF(tf::Transform m2o, ros::Time stamp)

        {
            assert(transformerSeted);
            MatrixWrapper::ColumnVector ll(3);
            decomposeTransform(trans_ptr->map2odom, ll);
            std::cout << "puser:" << ll << std::endl;
            trans_ptr->map2odom.setData(m2o);
            trans_ptr->map2odom.stamp_ = stamp;
            decomposeTransform(trans_ptr->map2odom, ll);
            std::cout << "puser:" << ll << std::endl;
        }

        bool addMeasurements(pose_factor::Ptr &factor)
        {
            factor_list.push_back(factor);
            if (factor_list.size() >= fuse_list_length)
            {
                sortList();
                fuseList();
                clearList();
            }
            std::cout << "reg" << factor_list.size() << std::endl;

            return true;
        }

        int getListLength(void)
        {
            return factor_list.size();
        }
    };
    class BR_pose_ekf
    {
    public:
        /// constructor
        BR_pose_ekf(int fuse_list_length = 5);

        // TODO 考虑置场请求
        void initialize(const tf::Transform &prior, const ros::Time &time);

        /// destructor
        virtual ~BR_pose_ekf();

    private:
        /// callback function for wheel_odom data
        void woCallback(const wheel_odomConstPtr &wheel_odom);

        /// callback function for visual_odom data
        void voCallback(const visual_odomConstPtr &visual_odom);

        // local map location by csm method
        void loCallback(const laser_odomConstPtr &laser_odom);
        void newloCallback(const laser_odomConstPtr &laser_odom);

        void getParams(void);

        // 根据参数来选择性初始化概率模型
        void createModels(void);

        // 根据params中的融合模式来初始化通讯相关对象
        void initTalkers(void);

        void decomposeTransform(const tf::Transform &trans, double &x, double &y, double &yaw);

        void decomposeTransform(const tf::Transform &trans, MatrixWrapper::ColumnVector &vec);

        // void angleOverflowCorrect(double &a, double ref);

        // static MatrixWrapper::SymmetricMatrix cov2wraped(boost::array<double, 36UL> cov);
        static MatrixWrapper::SymmetricMatrix cov2wraped(const boost::array<double, 9UL> &cov);

        // static boost::array<double, 36UL> wraped2cov(MatrixWrapper::SymmetricMatrix wraped);
        static boost::array<double, 9UL> wraped2cov(const MatrixWrapper::SymmetricMatrix &wraped);

        static boost::array<double, 9UL> downDim(const boost::array<double, 36UL> &dim6);

        static MatrixWrapper::SymmetricMatrix eigen2wraped(const Eigen::MatrixXd &eig);

        static Eigen::MatrixXd wraped2eigen(const MatrixWrapper::SymmetricMatrix &wraped);
        void truePoseCallback(const nav_msgs::Odometry::ConstPtr &odom_msg);
        void woCallback2(const wheel_odomConstPtr &odom);

        // 模式控制参数
        bool use_wo, use_vo, use_lo, use_true_pose;
        bool broadcastTF;
        bool odom_noise_test;
        bool predict_test;

        std::string map_frame;
        std::string base_footprint_frame;
        std::string laser_frame;
        std::string camera_frame;
        std::string odom_frame;
        std::string lo_topic;
        std::string wo_topic;
        std::string vo_topic;
        std::string output_topic;
        std::string true_pose_topic;
        std::string compensation_topic;


        double initPoseX, initPoseY, initPoseTheta;

        // 状态变量
        // bool wo_active_, imu_active_, vo_active_, gps_active_, lo_active_;
        bool filter_inited, wo_inited, vo_inited, lo_inited;
        bool received_true_pose_ = false;

        // 功能性变量
        MatrixWrapper::ColumnVector filter_estimate_old_vec_;
        tf::Transform filter_estimate_old_;
        ros::Time filter_time_old_;
        unsigned int wo_callback_counter_, imu_callback_counter_, lo_callback_counter_, vo_callback_counter_, gps_callback_counter_, ekf_sent_counter_;
        tf::Vector3 noise;
        ros::Time last_wo_time_, last_vo_time_, last_lo_time_;
        geometry_msgs::Twist last_wo_twist_;
        nav_msgs::Odometry latest_true_pose;
        nav_msgs::Odometry output_compensation;
        tf::Transform true_pose;
        boost::mutex lo_mutex;
        tf::Transform m2o_comp;
        pose_fuser fuser;

        // 通讯变量
        ros::NodeHandle node;
        ros::Publisher pose_pub;
        ros::Publisher compensation_pub;
        ros::Subscriber wo_sub, vo_sub, lo_sub, true_pose_sub;
        geometry_msgs::PoseWithCovarianceStamped output_;
        BR_transformer transformer;
        tf::TransformBroadcaster pose_broadcaster_;

        // 滤波器相关模型和变量  pdf / model / ekf filter
        BFL::LinearAnalyticSystemModelGaussianUncertainty *sys_model_;
        BFL::LinearAnalyticConditionalGaussian *sys_pdf_;
        BFL::LinearAnalyticSystemModelGaussianUncertainty *tmp_sys_model_;
        BFL::LinearAnalyticConditionalGaussian *tmp_sys_pdf_;

        BFL::LinearAnalyticConditionalGaussian *lo_meas_pdf_;
        BFL::LinearAnalyticMeasurementModelGaussianUncertainty *lo_meas_model_;
        BFL::LinearAnalyticConditionalGaussian *wo_meas_pdf_;
        BFL::LinearAnalyticMeasurementModelGaussianUncertainty *wo_meas_model_;
        BFL::LinearAnalyticConditionalGaussian *vo_meas_pdf_;
        BFL::LinearAnalyticMeasurementModelGaussianUncertainty *vo_meas_model_;
        BFL::Gaussian *prior_;
        BFL::ExtendedKalmanFilter *filter_;
        BFL::ExtendedKalmanFilter *tmpfilter_;
        MatrixWrapper::SymmetricMatrix wo_covariance_, imu_covariance_, vo_covariance_, gps_covariance_, lo_covariance_;

        // 后续将淘汰
        tf::Transform wo_meas_, imu_meas_, vo_meas_, lo_meas_;
        tf::Transform base_vo_init_;
        tf::StampedTransform camera_base_;

        ros::Time wo_stamp_, imu_stamp_, vo_stamp_, gps_stamp_, filter_stamp_, lo_stamp_;
        ros::Time wo_init_stamp_, imu_init_stamp_, vo_init_stamp_, gps_init_stamp_, lo_init_stamp_;

        double timeout_;
        bool debug_, self_diagnose_;

        // log files for debugging
        std::ofstream wo_file_, imu_file_, vo_file_, gps_file_, corr_file_, time_file_, extra_file_;

    }; // class

}; // namespace

#endif
