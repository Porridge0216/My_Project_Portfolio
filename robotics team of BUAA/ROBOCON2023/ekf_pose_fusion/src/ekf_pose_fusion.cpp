#include "ekf_pose_fusion/ekf_pose_fusion.hpp"
#include <thread>
#include <random>
using namespace MatrixWrapper;
using namespace BFL;
using namespace tf;
using namespace std;
using namespace ros;
// using namespace uncertain_tf;

namespace estimation
{
    BR_pose_ekf::BR_pose_ekf(int fuse_list_length)
        : prior_(NULL),
          filter_(NULL),
          filter_inited(false),
          wo_inited(false),
          vo_inited(false),
          lo_inited(false)

    {
        getParams();
        fuse_list_length = fuse_list_length;
        // create SYSTEM MODEL
        // TODO 系统协方差调整
        // 并找到控制量输入的窗口
        // 思考应当将速度测量量作为控制量还是观测
        ColumnVector sysNoise_Mu(3);
        sysNoise_Mu = 0;
        SymmetricMatrix sysNoise_Cov(3);
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

        vector<Matrix> AB(2);
        AB[0] = A;
        AB[1] = B;

        for (unsigned int i = 1; i <= 2; i++)
            sysNoise_Cov(i, i) = pow(0.03, 2);
        sysNoise_Cov(3, 3) = pow(0.11, 2);

        Gaussian system_Uncertainty(sysNoise_Mu, sysNoise_Cov);
        sys_pdf_ = new LinearAnalyticConditionalGaussian(AB, system_Uncertainty);
        sys_model_ = new LinearAnalyticSystemModelGaussianUncertainty(sys_pdf_);
        tmp_sys_pdf_ = new LinearAnalyticConditionalGaussian(AB, system_Uncertainty);
        tmp_sys_model_ = new LinearAnalyticSystemModelGaussianUncertainty(tmp_sys_pdf_);

        // create MEASUREMENT MODEL ODOM
        // wheel_odom概率模型创建
        ColumnVector measNoiseOdom_Mu(3);
        measNoiseOdom_Mu = 0;
        SymmetricMatrix measNoiseOdom_Cov(3);
        measNoiseOdom_Cov = 0;
        for (unsigned int i = 1; i <= 3; i++)
            measNoiseOdom_Cov(i, i) = pow(0.008, 2);
        Gaussian measurement_Uncertainty_Odom(measNoiseOdom_Mu, measNoiseOdom_Cov);
        Matrix Hodom(3, 3);
        Hodom = 0;
        Hodom(1, 1) = 1;
        Hodom(2, 2) = 1;
        Hodom(3, 3) = 1;
        wo_meas_pdf_ = new LinearAnalyticConditionalGaussian(Hodom, measurement_Uncertainty_Odom);
        wo_meas_model_ = new LinearAnalyticMeasurementModelGaussianUncertainty(wo_meas_pdf_);

        // create MEASUREMENT MODEL VO
        // 视觉定位数据
        ColumnVector measNoiseVo_Mu(3);
        measNoiseVo_Mu = 0;
        SymmetricMatrix measNoiseVo_Cov(3);
        measNoiseVo_Cov = 0;
        for (unsigned int i = 1; i <= 3; i++)
            measNoiseVo_Cov(i, i) = 1;
        Gaussian measurement_Uncertainty_Vo(measNoiseVo_Mu, measNoiseVo_Cov);
        Matrix Hvo(3, 3);
        Hvo = 0;
        Hvo(1, 1) = 1;
        Hvo(2, 2) = 1;
        Hvo(3, 3) = 1;

        vo_meas_pdf_ = new LinearAnalyticConditionalGaussian(Hvo, measurement_Uncertainty_Vo);
        vo_meas_model_ = new LinearAnalyticMeasurementModelGaussianUncertainty(vo_meas_pdf_);

        // create MEASUREMENT MODEL LO
        ColumnVector measNoiseLo_Mu(3);
        measNoiseLo_Mu = 0;
        SymmetricMatrix measNoiseLo_Cov(3);
        measNoiseLo_Cov = 0;
        for (unsigned int i = 1; i <= 2; i++)
            measNoiseLo_Cov(i, i) = pow(0.001, 2);
        measNoiseLo_Cov(3, 3) = 9e-7;
        Gaussian measurement_Uncertainty_Lo(measNoiseLo_Mu, measNoiseLo_Cov);
        Matrix Hlo(3, 3);
        Hlo = 0;
        Hlo(1, 1) = 1;
        Hlo(2, 2) = 1;
        Hlo(3, 3) = 1;

        Eigen::MatrixXd defult_cov;
        lo_meas_pdf_ = new LinearAnalyticConditionalGaussian(Hlo, measurement_Uncertainty_Lo);
        lo_meas_model_ = new LinearAnalyticMeasurementModelGaussianUncertainty(lo_meas_pdf_);

        transformer.set_m2o(0, 0, 0, Time::now());
        transformer.set_o2b(initPoseX, initPoseY, initPoseTheta, Time::now());
        transformer.set_m2b_cov(defult_cov, Time::now());

        initTalkers();
        initialize(transformer.map2odom * transformer.odom2basefootprint, Time::now());
        fuser.setTransformer(&transformer);
        fuser.initFilter(transformer.map2odom * transformer.odom2basefootprint, Time::now());
        transformer.setFuser(&fuser);
    }

    BR_pose_ekf::~BR_pose_ekf()
    {
        if (filter_)
            delete filter_;
        if (prior_)
            delete prior_;

        if (use_wo)
        {
            delete wo_meas_model_;
            delete wo_meas_pdf_;
        }
        if (use_vo)
        {
            delete vo_meas_model_;
            delete vo_meas_pdf_;
        }

        if (use_lo)
        {
            delete lo_meas_model_;
            delete lo_meas_pdf_;
        }

        delete sys_pdf_;
        delete sys_model_;
    };

    void BR_pose_ekf::initTalkers(void)
    {
        if (use_lo)
        {

            // lo_sub = node.subscribe(
            //     lo_topic, 1, &BR_pose_ekf::loCallback, this);

            ros::CallbackQueue *cbque = new ros::CallbackQueue();

            ros::SubscribeOptions ops =
                ros::SubscribeOptions::create<geometry_msgs::PoseWithCovarianceStamped>(
                    lo_topic,                                           // topic name
                    1,                                                  // queue length
                    boost::bind(&BR_pose_ekf::newloCallback, this, _1), // callback
                    ros::VoidPtr(),                                     // tracked object, we don't need one thus NULL
                    cbque);
            ops.allow_concurrent_callbacks = false;
            ops.transport_hints = ros::TransportHints().tcpNoDelay().udp();
            // // subscribe
            lo_sub = node.subscribe(ops);

            // spawn async spinner with 1 thread, running on our custom queue
            ros::AsyncSpinner *async_spinner = new ros::AsyncSpinner(1, cbque);

            // start the spinner
            async_spinner->start();
        }
        if (use_vo)
        {
            vo_sub = node.subscribe(
                vo_topic, 1, &BR_pose_ekf::voCallback, this);
        }
        if (use_wo)
        {
            // wo_sub = node.subscribe(
            //     wo_topic, 1, &BR_pose_ekf::woCallback2, this);

            // ros::SubscribeOptions ops;
            // ops.init<nav_msgs::Odometry>(wo_topic, 1, boost::bind(&BR_pose_ekf::woCallback, this, _1));
            // ops.allow_concurrent_callbacks = true;
            // wo_sub = node.subscribe(ops);

            ros::CallbackQueue *cbque = new ros::CallbackQueue();

            ros::SubscribeOptions ops =
                ros::SubscribeOptions::create<nav_msgs::Odometry>(
                    wo_topic,                                         // topic name
                    1,                                                // queue length
                    boost::bind(&BR_pose_ekf::woCallback2, this, _1), // callback
                    ros::VoidPtr(),                                   // tracked object, we don't need one thus NULL
                    cbque);
            // // subscribe
            ops.allow_concurrent_callbacks = true;
            ops.transport_hints = ros::TransportHints().tcpNoDelay().udp();

            wo_sub = node.subscribe(ops);

            ros::AsyncSpinner *async_spinner = new ros::AsyncSpinner(1, cbque);
            async_spinner->start();
        }
        if (use_true_pose)
        {
            true_pose_sub = node.subscribe(
                true_pose_topic, 1, &BR_pose_ekf::truePoseCallback, this, ros::TransportHints().tcpNoDelay().udp());
        }

        pose_pub = node.advertise<geometry_msgs::PoseWithCovarianceStamped>(output_topic, 1);
        compensation_pub = node.advertise<nav_msgs::Odometry>(compensation_topic, 1);
    }

    void BR_pose_ekf::truePoseCallback(const nav_msgs::Odometry::ConstPtr &odom_msg)
    {
        assert(use_true_pose);
        latest_true_pose = *odom_msg;
        tf::poseMsgToTF(latest_true_pose.pose.pose, true_pose);
        transformer.true_pose_list.insertData(tf::TransformStorage(tf::StampedTransform(true_pose, odom_msg->header.stamp, odom_msg->header.frame_id, odom_msg->child_frame_id), 1, 3));
        if (!received_true_pose_)
        {
            received_true_pose_ = true;
        }

        std::cout << "lo** ined" << std::endl;
        if (odom_msg->header.stamp < fuser.last_filt_time)
            return;

        lo_stamp_ = odom_msg->header.stamp;
        poseMsgToTF(odom_msg->pose.pose, lo_meas_);
        MatrixWrapper::ColumnVector noise(3);
        tf::Transform noitf;
        std::default_random_engine generator(rand());
        std::normal_distribution<double> distribution(0.0, 0.03);
        noise(1) = distribution(generator);
        noise(2) = distribution(generator);
        noise(3) = distribution(generator);
        cout << "noise: " << noise << endl;
        ColumnVector2Transform(noise, noitf);
        lo_meas_ *= noitf;

        TransformStorage o2b, tp;
        if (!transformer.o2b_list.getData(lo_stamp_, o2b))
        {
            transformer.o2b_list.getData(Time(0), o2b);
            ROS_ERROR("no lo stamp find");
        };

        if (!transformer.true_pose_list.getData(lo_stamp_, tp))
        {
            transformer.true_pose_list.getData(Time(0), tp);
        }

        tf::Transform o2bTran(o2b.rotation_, o2b.translation_);
        tf::Transform tpTran(tp.rotation_, tp.translation_);

        pose_factor::Ptr lo_factor(new pose_factor());
        lo_factor->stamp = lo_stamp_;
        lo_factor->woTrans = o2bTran;
        lo_factor->cov = Eigen::Matrix3d::Identity() * 3e-7;

        lo_factor->measurement = lo_meas_;
        fuser.addMeasurements(lo_factor);

        // poseTFToMsg(lo_meas_, output_.pose.pose);
        // pose_pub.publish(output_);
        // if (broadcastTF)
        // {
        //     pose_broadcaster_.sendTransform(StampedTransform(transformer.map2odom, odom->header.stamp, map_frame, odom_frame));
        //     pose_broadcaster_.sendTransform(StampedTransform(transformer.odom2basefootprint, odom->header.stamp, odom_frame, base_footprint_frame));
        // }
    }

    void BR_pose_ekf::getParams(void)
    {
        ros::NodeHandle n_pri("~");
        // n_pri.param<std::string>("map_file_path", map_file_path, "/home/scy/WorkSpaces/myTools/src/laser_mapper/maps/location_test_map.pcd");
        n_pri.param<bool>("use_wo", use_wo, true);
        n_pri.param<bool>("use_vo", use_vo, false);
        n_pri.param<bool>("use_lo", use_lo, false);
        n_pri.param<bool>("use_true_pose", use_true_pose, false);
        n_pri.param<bool>("predict_test", predict_test, false);
        n_pri.param<bool>("broadcastTF", broadcastTF, true);
        n_pri.param<bool>("odom_noise_test", odom_noise_test, false);

        n_pri.param<string>("map_frame", map_frame, "map");
        n_pri.param<string>("base_footprint_frame", base_footprint_frame, "base_footprint");
        n_pri.param<string>("lo_topic", lo_topic, "laser_odom");
        n_pri.param<string>("wo_topic", wo_topic, "wheel_odom");
        n_pri.param<string>("vo_topic", vo_topic, "visual_odom");
        n_pri.param<string>("true_pose_topic", true_pose_topic, "/ground_truth/state");
        // 待会儿看看fuse之前的先验值是不是odom值，输入值是不是对
        n_pri.param<string>("output_topic", output_topic, "BR_Pose");
        n_pri.param<string>("laser_frame", laser_frame, "laser");
        n_pri.param<string>("camera_frame", camera_frame, "camera");
        n_pri.param<string>("odom_frame", odom_frame, "odom");
        n_pri.param<double>("initPoseX", initPoseX, 5.0);
        n_pri.param<double>("initPoseY", initPoseY, 5.0);
        n_pri.param<double>("initPoseTheta", initPoseTheta, 5.0);
	    n_pri.param<std::string>("compensation_topic", compensation_topic, "/compensation");

    }

    // initialize prior density of filter
    void BR_pose_ekf::initialize(const Transform &prior, const Time &time)
    {
        // set prior of filter
        ColumnVector prior_Mu(3);
        decomposeTransform(prior, prior_Mu(1), prior_Mu(2), prior_Mu(3));
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
        auto prior2_ = new Gaussian(prior_Mu, prior_Cov);
        tmpfilter_ = new ExtendedKalmanFilter(prior2_);

        // remember prior
        // addMeasurement(StampedTransform(prior, time, output_frame_, base_footprint_frame_));
        filter_estimate_old_vec_ = prior_Mu;
        filter_estimate_old_ = prior;
        filter_time_old_ = time;

        transformer.set_m2b_cov(wraped2eigen(prior_Cov), time);

        // filter initialized
        filter_inited = true;

        // TODO 初始化tf树位姿
    }

    // decompose Transform into x,y,z,Rx,Ry,Rz
    void BR_pose_ekf::decomposeTransform(const Transform &trans, double &x, double &y, double &yaw)
    {
        x = trans.getOrigin().x();
        y = trans.getOrigin().y();
        yaw = getYaw(trans.getRotation());
    };
    void BR_pose_ekf::decomposeTransform(const Transform &trans, ColumnVector &vec)
    {
        vec(1) = trans.getOrigin().x();
        vec(2) = trans.getOrigin().y();
        vec(3) = getYaw(trans.getRotation());
    };

    // correct for angle overflow

    SymmetricMatrix BR_pose_ekf::cov2wraped(const boost::array<double, 9UL> &cov)
    {
        SymmetricMatrix ret(3);

        for (unsigned int i = 0; i < 3; i++)
            for (unsigned int j = 0; j < 3; j++)
                ret(i + 1, j + 1) = cov[3 * i + j];

        return ret;
    }
    boost::array<double, 9UL> BR_pose_ekf::wraped2cov(const SymmetricMatrix &wraped)
    {
        boost::array<double, 9UL> ret;
        for (unsigned int i = 0; i < 3; i++)
            for (unsigned int j = 0; j < 3; j++)
                ret[3 * i + j] = wraped(i + 1, j + 1);

        return ret;
    }
    boost::array<double, 9UL> BR_pose_ekf::downDim(const boost::array<double, 36UL> &dim6)
    {
        boost::array<double, 9UL> ret;
        for (unsigned int i = 0; i < 2; i++)
            for (unsigned int j = 0; j < 2; j++)
                ret[3 * i + j] = dim6[6 * i + j];

        ret[6] = dim6[30];
        ret[7] = dim6[31];
        ret[8] = dim6[35];
        ret[2] = dim6[5];
        ret[5] = dim6[11];
        return ret;
    }

    MatrixWrapper::SymmetricMatrix BR_pose_ekf::eigen2wraped(const Eigen::MatrixXd &eig)
    {
        MatrixWrapper::SymmetricMatrix ret(eig.rows());
        for (unsigned int i = 0; i < eig.rows(); i++)
            for (unsigned int j = 0; j < eig.rows(); j++)
                ret(i + 1, j + 1) = eig(i, j);
        return ret;
    }

    void BR_pose_ekf::newloCallback(const laser_odomConstPtr &lo)
    {
        boost::mutex::scoped_lock(lo_mutex);

        assert(use_lo);
        std::cout << "lo** ined" << std::endl;
        if (lo->header.stamp < fuser.last_filt_time)
            return;

        lo_stamp_ = lo->header.stamp;
        poseMsgToTF(lo->pose.pose, lo_meas_);

        TransformStorage o2b, tp;
        if (!transformer.o2b_list.getData(lo_stamp_, o2b))
        {
            transformer.o2b_list.getData(Time(0), o2b);
            ROS_ERROR("no lo stamp find");
        };

        if (!transformer.true_pose_list.getData(lo_stamp_, tp))
        {
            transformer.true_pose_list.getData(Time(0), tp);
        }

        tf::Transform o2bTran(o2b.rotation_, o2b.translation_);
        tf::Transform tpTran(tp.rotation_, tp.translation_);

        pose_factor::Ptr lo_factor(new pose_factor());
        lo_factor->stamp = lo_stamp_;
        lo_factor->woTrans = o2bTran;
        // TODO
        lo_factor->cov = wraped2eigen(cov2wraped(downDim(lo->pose.covariance)));      
        std::cout<<"lo_factor->cov"<<lo_factor->cov<<std::endl;
        // lo_factor->cov = Eigen::Matrix3d::Identity();

        lo_factor->measurement = lo_meas_;
        fuser.addMeasurements(lo_factor);

        // ColumnVector meas_vec(3);
        // decomposeTransform(lo_meas_, meas_vec);
        // ColumnVector pri_vec(3);
        // angleOverflowCorrect(meas_vec(3), pri_vec(3));

        // // TODO
        // // 获取stamp的位姿、协方差
        // // filter_estimate_old_vec_应该改成获取的位姿，检查溢出
        // // 新的filter中赋值位姿与协方差作为先验
        // // 用lo_meas融合出新的位姿后验值
        // // 新m2o=旧m2o*纠错值
        // // 新cov=  ？
        // // 对m2o进行#当前stamp#的更新，并发布（悟已往之不谏，知来者之可追，对lo stamp之后的所有m2o都进行纠正是不现实的）
        // // 后续还可以检测一下这之间时差多少（lo stamp 与 now())
        // // 可以只要cache？两个cache维护，发布只发布最新的即可？

        // // update filter
        // MatrixWrapper::ColumnVector tmpvec(3);

        // decomposeTransform(tpTran, tmpvec);
        // cout << "tpvec: " << tmpvec << endl;
        // cout << "lo pri: " << pri_vec << endl;

        // tmpfilter_->PostGet()->ExpectedValueSet(pri_vec);
        // // // TODO 需要考虑滤波器的设计，状态数量，以及记得对lo数据进行tf，变到base_footprint下
        // tmpfilter_->PostGet()->CovarianceSet(eigen2wraped(priCov.covariance_));
        // // cout << "lastvec: " << tmpfilter_->PostGet()->ExpectedValueGet() << endl;
        // boost::array<double, 9UL> cov3x3 = downDim(lo->pose.covariance);
        // // cout << cov2wraped(cov3x3) << endl;
        // // lo_meas_pdf_->AdditiveNoiseSigmaSet(cov2wraped(cov3x3));
        // angleOverflowCorrect(meas_vec(3), tmpfilter_->PostGet()->ExpectedValueGet()(3));
        // cout << "filter cov: " << tmpfilter_->PostGet()->CovarianceGet() << endl;
        // cout << "sys cov: " << sys_model_->SystemPdfGet()->CovarianceGet() << endl;
        // cout << "lo cov: " << lo_meas_model_->MeasurementPdfGet()->CovarianceGet() << endl;

        // tmpfilter_->Update(sys_model_, MatrixWrapper::ColumnVector(3) = 0, lo_meas_model_, meas_vec);
        // customizAngle_in_fabsPi(tmpfilter_->PostGet()->ExpectedValueGet()(3));
        // auto post_vec = tmpfilter_->PostGet()->ExpectedValueGet();
        // auto post_cov = tmpfilter_->PostGet()->CovarianceGet();

        // ColumnVector2Transform(post_vec, postTran);
        // cout << "laser odom: " << meas_vec << endl;
        // cout << "lo post: " << post_vec << endl;

        // // output_.header.frame_id = map_frame;
        // // output_.header.stamp = lo_stamp_;
        // // // output_.pose.covariance = wraped2cov(filter_->PostGet()->CovarianceGet());
        // // poseTFToMsg(lo_meas_, output_.pose.pose);
        // // pose_pub.publish(output_);

        // // decomposeTransform(m2oTran * o2bTran * priTran.inverse() * postTran * o2bTran.inverse(), tmpvec);

        // // ROS_ERROR_STREAM("comp: " << tmpvec << endl);
        // // decomposeTransform(m2oTran * o2bTran * priTran.inverse() * postTran, tmpvec);
        // // ROS_ERROR_STREAM("m2b: " << tmpvec << endl);

        // transformer.set_m2o(postTran * o2bTran.inverse(), ros::Time::now());
        // // transformer.set_m2o(m2oTran * o2bTran * priTran.inverse() * lo_meas_ * o2bTran.inverse(), ros::Time::now());

        // decomposeTransform(transformer.map2odom * o2bTran, meas_vec);
        // cout << "corrected : " << meas_vec << endl;
        // // decomposeTransform(lo_meas_, post_vec);

        // auto err = (post_vec - meas_vec);
        // ROS_WARN_STREAM_COND(fabs(err(1)) + fabs(err(2)) > 1e-4, "err too much: " << err);
        // // tf::Quaternion q;
        // // q.setRPY(0, 0, filter_estimate_old_vec_(3));
        // // filter_estimate_old_ = tf::Transform(q, Vector3(filter_estimate_old_vec_(1), filter_estimate_old_vec_(2), 0));
        // // filter_time_old_ = lo_stamp_;
        // ROS_INFO_STREAM("lo out: " << ros::Time::now() << endl);
    }
    void BR_pose_ekf::loCallback(const laser_odomConstPtr &lo)
    {
        boost::mutex::scoped_lock(lo_mutex);

        assert(use_lo);
        while (filter_time_old_ < lo->header.stamp)
        {
            ros::Duration rater;
            rater.fromSec(0.005);
            rater.sleep();
        }
        ROS_INFO_STREAM("lo in: " << ros::Time::now() << endl
                                  << "stamp: " << lo->header.stamp);
        lo_callback_counter_++;
        lo_stamp_ = lo->header.stamp;
        poseMsgToTF(lo->pose.pose, lo_meas_);
        TransformStorage m2o, o2b, tp;
        if ((!transformer.m2o_list.getData(lo_stamp_, m2o)) || (!transformer.o2b_list.getData(lo_stamp_, o2b)))
        {
            transformer.m2o_list.getData(Time(0), m2o);
            transformer.o2b_list.getData(Time(0), o2b);
            ROS_ERROR("no lo stamp find");
        };
        if (!transformer.true_pose_list.getData(lo_stamp_, tp))
        {
            transformer.true_pose_list.getData(Time(0), tp);
        }
        // if (!filter_inited)
        // {
        //     // init filter
        //     initialize(lo_meas_, lo->header.stamp);
        // }

        // TODO sensors_inited 到底有没有用处
        if (!lo_inited)
        {
            last_lo_time_ = lo->header.stamp;
            lo_inited = true;
        }

        tf::Transform m2oTran(transformer.map2odom), o2bTran(o2b.rotation_, o2b.translation_); // 原来是 m2oTran(m2o.rotation_, m2o.translation_) ，为了测试使用新的m2o而不是根据时间戳
        tf::Transform priTran, tpTran(tp.rotation_, tp.translation_);
        tf::Transform postTran;
        uncertain_tf::CovarianceStorage priCov;
        transformer.lookupPoseAndCov(lo_stamp_, priTran, priCov);

        ColumnVector meas_vec(3);
        decomposeTransform(lo_meas_, meas_vec);
        ColumnVector pri_vec(3);
        decomposeTransform(priTran, pri_vec);
        angleOverflowCorrect(meas_vec(3), pri_vec(3));

        // TODO
        // 获取stamp的位姿、协方差
        // filter_estimate_old_vec_应该改成获取的位姿，检查溢出
        // 新的filter中赋值位姿与协方差作为先验
        // 用lo_meas融合出新的位姿后验值
        // 新m2o=旧m2o*纠错值
        // 新cov=  ？
        // 对m2o进行#当前stamp#的更新，并发布（悟已往之不谏，知来者之可追，对lo stamp之后的所有m2o都进行纠正是不现实的）
        // 后续还可以检测一下这之间时差多少（lo stamp 与 now())
        // 可以只要cache？两个cache维护，发布只发布最新的即可？

        // update filter
        MatrixWrapper::ColumnVector tmpvec(3);

        decomposeTransform(tpTran, tmpvec);
        cout << "tpvec: " << tmpvec << endl;
        cout << "lo pri: " << pri_vec << endl;

        tmpfilter_->PostGet()->ExpectedValueSet(pri_vec);
        // // TODO 需要考虑滤波器的设计，状态数量，以及记得对lo数据进行tf，变到base_footprint下
        tmpfilter_->PostGet()->CovarianceSet(eigen2wraped(priCov.covariance_));
        // cout << "lastvec: " << tmpfilter_->PostGet()->ExpectedValueGet() << endl;
        boost::array<double, 9UL> cov3x3 = downDim(lo->pose.covariance);
        // cout << cov2wraped(cov3x3) << endl;
        // lo_meas_pdf_->AdditiveNoiseSigmaSet(cov2wraped(cov3x3));
        angleOverflowCorrect(meas_vec(3), tmpfilter_->PostGet()->ExpectedValueGet()(3));
        cout << "filter cov: " << tmpfilter_->PostGet()->CovarianceGet() << endl;
        cout << "sys cov: " << sys_model_->SystemPdfGet()->CovarianceGet() << endl;
        cout << "lo cov: " << lo_meas_model_->MeasurementPdfGet()->CovarianceGet() << endl;

        tmpfilter_->Update(sys_model_, MatrixWrapper::ColumnVector(3) = 0, lo_meas_model_, meas_vec);
        customizAngle_in_fabsPi(tmpfilter_->PostGet()->ExpectedValueGet()(3));
        auto post_vec = tmpfilter_->PostGet()->ExpectedValueGet();
        auto post_cov = tmpfilter_->PostGet()->CovarianceGet();

        ColumnVector2Transform(post_vec, postTran);
        cout << "laser odom: " << meas_vec << endl;
        cout << "lo post: " << post_vec << endl;

        // output_.header.frame_id = map_frame;
        // output_.header.stamp = lo_stamp_;
        // // output_.pose.covariance = wraped2cov(filter_->PostGet()->CovarianceGet());
        // poseTFToMsg(lo_meas_, output_.pose.pose);
        // pose_pub.publish(output_);

        // decomposeTransform(m2oTran * o2bTran * priTran.inverse() * postTran * o2bTran.inverse(), tmpvec);

        // ROS_ERROR_STREAM("comp: " << tmpvec << endl);
        // decomposeTransform(m2oTran * o2bTran * priTran.inverse() * postTran, tmpvec);
        // ROS_ERROR_STREAM("m2b: " << tmpvec << endl);

        transformer.set_m2o(postTran * o2bTran.inverse(), ros::Time::now());
        // transformer.set_m2o(m2oTran * o2bTran * priTran.inverse() * lo_meas_ * o2bTran.inverse(), ros::Time::now());

        decomposeTransform(transformer.map2odom * o2bTran, meas_vec);
        cout << "corrected : " << meas_vec << endl;
        // decomposeTransform(lo_meas_, post_vec);

        auto err = (post_vec - meas_vec);
        ROS_WARN_STREAM_COND(fabs(err(1)) + fabs(err(2)) > 1e-4, "err too much: " << err);
        // tf::Quaternion q;
        // q.setRPY(0, 0, filter_estimate_old_vec_(3));
        // filter_estimate_old_ = tf::Transform(q, Vector3(filter_estimate_old_vec_(1), filter_estimate_old_vec_(2), 0));
        // filter_time_old_ = lo_stamp_;
        ROS_INFO_STREAM("lo out: " << ros::Time::now() << endl);
    }

    Eigen::MatrixXd BR_pose_ekf::wraped2eigen(const MatrixWrapper::SymmetricMatrix &wraped)
    {
        Eigen::MatrixXd ret;
        ret = Eigen::MatrixXd::Zero(wraped.size1(), wraped.size1());
        for (unsigned int i = 0; i < ret.rows(); i++)
            for (unsigned int j = 0; j < ret.rows(); j++)
                ret(i, j) = wraped(i + 1, j + 1);

        return ret;
    }

    void BR_pose_ekf::voCallback(const visual_odomConstPtr &vo)
    {
        assert(use_vo);
        vo_stamp_ = vo->header.stamp;
        poseMsgToTF(vo->pose.pose, vo_meas_);
        TransformStorage m2o, o2b;
        if ((!transformer.m2o_list.getData(vo_stamp_, m2o)) || (!transformer.o2b_list.getData(vo_stamp_, o2b)))
        {
            transformer.m2o_list.getData(Time(0), m2o);
            transformer.o2b_list.getData(Time(0), o2b);
            ROS_ERROR("no wo stamp find");
        };
        tf::Transform m2oTran(m2o.rotation_, m2o.translation_), o2bTran(o2b.rotation_, o2b.translation_);
        tf::Transform priTran;
        tf::Transform postTran;
        uncertain_tf::CovarianceStorage priCov;
        transformer.lookupPoseAndCov(vo_stamp_, priTran, priCov);
        ColumnVector meas_vec(3);
        decomposeTransform(vo_meas_, meas_vec);
        ColumnVector pri_vec(3);
        decomposeTransform(priTran, pri_vec);
        angleOverflowCorrect(meas_vec(3), pri_vec(3));

        tmpfilter_->PostGet()->ExpectedValueSet(pri_vec);
        // TODO 需要考虑滤波器的设计，状态数量，以及记得对lo数据进行tf，变到base_footprint下
        tmpfilter_->PostGet()->CovarianceSet(eigen2wraped(priCov.covariance_));

        boost::array<double, 9UL> cov3x3 = downDim(vo->pose.covariance);
        vo_meas_pdf_->AdditiveNoiseSigmaSet(cov2wraped(cov3x3));
        tmpfilter_->Update(sys_model_, MatrixWrapper::ColumnVector(3) = 0, vo_meas_model_, meas_vec);

        auto post_vec = filter_->PostGet()->ExpectedValueGet();
        auto post_cov = filter_->PostGet()->CovarianceGet();
        ColumnVector2Transform(post_vec, postTran);

        // pri * corr = post;
        // m2o1 * comp * o2b = m2o2 * o2b;
        // m2o1 * o2b * corr = m2o2 * o2b

        transformer.compensate_m2o(o2bTran * priTran.inverse() * postTran * o2bTran.inverse(), ros::Time::now());
        // transformer.set_m2b(newPose(1), newPose(2), newPose(6), );
        // tf::Quaternion q;
        // q.setRPY(0, 0, filter_estimate_old_vec_(3));
        // filter_estimate_old_ = tf::Transform(q, Vector3(filter_estimate_old_vec_(1), filter_estimate_old_vec_(2), 0));

        // filter_time_old_ = lo->header.stamp;
    };
    void BR_pose_ekf::woCallback2(const wheel_odomConstPtr &odom)
    {
        assert(use_wo);
        ROS_INFO_STREAM("wo in**: " << ros::Time::now() << endl
                                  << "stamp: " << odom->header.stamp);

        tf::poseMsgToTF(odom->pose.pose, wo_meas_);
        auto tr = wo_meas_.getOrigin();
        tr.setZ(0);
        wo_meas_.setOrigin(tr);
        MatrixWrapper::ColumnVector odom_vec(3);
        wo_stamp_ = odom->header.stamp;
        tf::Transform curr_m2o = transformer.map2odom;
        // transformer.set_m2o(transformer.map2odom, wo_stamp_);
        // transformer.set_o2b(m2otf.inverse() * filter_estimate_old_, wo_stamp_);
        transformer.set_o2b(wo_meas_, wo_stamp_);

        // transformer.set_m2b_cov(wraped2eigen(cov2wraped(downDim(odom->pose.covariance))), wo_stamp_);
        MatrixWrapper::SymmetricMatrix wo_cov(3);
        for (unsigned int i = 1; i <= 2; i++)
            wo_cov(i, i) = pow(0.16, 2);
        wo_cov(3, 3) = 9e-2;
        transformer.set_m2b_cov(wraped2eigen(wo_cov), wo_stamp_);

        filter_time_old_ = wo_stamp_;

        // output_.pose.covariance = wraped2cov(filter_->PostGet()->CovarianceGet());
        MatrixWrapper::ColumnVector ll(3);
        decomposeTransform(curr_m2o, ll);
        std::cout << "wo2zhong" << ll << std::endl;
        filter_estimate_old_ = curr_m2o * wo_meas_;
        decomposeTransform(filter_estimate_old_, odom_vec);

        angleOverflowCorrect(odom_vec(3), filter_->PostGet()->ExpectedValueGet()(3));

        filter_->Update(sys_model_, ColumnVector(3) = 0, wo_meas_model_, odom_vec);
        // cout << "wo filter cov: " << filter_->PostGet()->CovarianceGet() << endl;
        // cout << "wo sys cov: " << sys_model_->SystemPdfGet()->CovarianceGet() << endl;
        // cout << "wo lo cov: " << wo_meas_model_->MeasurementPdfGet()->CovarianceGet() << endl;

        customizAngle_in_fabsPi(filter_->PostGet()->ExpectedValueGet()(3));
        auto post_vec = filter_->PostGet()->ExpectedValueGet();
        auto post_cov = filter_->PostGet()->CovarianceGet();
        double ot = tf::getYaw(latest_true_pose.pose.pose.orientation);
        angleOverflowCorrect(ot, post_vec(3));
        // ColumnVector2Transform(post_vec, filter_estimate_old_);

        output_.header.frame_id = map_frame;
        output_.header.stamp = odom->header.stamp;
        poseTFToMsg(filter_estimate_old_, output_.pose.pose);
        pose_pub.publish(output_);

        output_compensation.header.frame_id = map_frame;
        output_compensation.header.stamp = odom->header.stamp;
        poseTFToMsg(transformer.map2odom, output_compensation.pose.pose);
        compensation_pub.publish(output_compensation);

        if (broadcastTF)
        {
            pose_broadcaster_.sendTransform(StampedTransform(transformer.map2odom, odom->header.stamp, map_frame, odom_frame));
            pose_broadcaster_.sendTransform(StampedTransform(transformer.odom2basefootprint, odom->header.stamp, odom_frame, base_footprint_frame));
        }
    }

    void BR_pose_ekf::woCallback(const wheel_odomConstPtr &odom)
    {
        assert(use_wo);

        ROS_INFO_STREAM("wo in: " << ros::Time::now() << endl
                                  << "stamp: " << odom->header.stamp);

        wo_callback_counter_++;

        tf::poseMsgToTF(odom->pose.pose, wo_meas_);
        wo_stamp_ = odom->header.stamp;
        TransformStorage m2o, o2b, tp;
        if ((!transformer.m2o_list.getData(wo_stamp_, m2o)) || (!transformer.o2b_list.getData(wo_stamp_, o2b)))
        {
            transformer.m2o_list.getData(Time(0), m2o);
            transformer.o2b_list.getData(Time(0), o2b);
        };
        if (!transformer.true_pose_list.getData(wo_stamp_, tp))
        {
            transformer.true_pose_list.getData(Time(0), tp);
            ROS_ERROR("true pose not find");
        }
        ColumnVector m2ovec(3);
        // decomposeTransform(Transform(m2o.rotation_, m2o.translation_), m2ovec);
        // cout << "m2ovec: " << m2ovec << endl;
        ColumnVector o2bvec(3);
        // decomposeTransform(Transform(o2b.rotation_, o2b.translation_), o2bvec);
        // cout << "o2bvec: " << o2bvec << endl;
        // ColumnVector tpvec(3);

        decomposeTransform(tf::Transform(tp.rotation_, tp.translation_), o2bvec);
        cout << "tpvec: " << o2bvec << endl;

        // if (!filter_inited)
        // {
        //     // init filter
        //     initialize(wo_meas_, odom->header.stamp);
        // }
        if (!wo_inited)
        {
            last_wo_time_ = odom->header.stamp;
            last_wo_twist_ = odom->twist.twist;
            wo_inited = true;
        }

        ColumnVector odom_rel(3);
        decomposeTransform(Transform(m2o.rotation_, m2o.translation_) * wo_meas_, odom_rel);
        angleOverflowCorrect(odom_rel(3), filter_estimate_old_vec_(3));
        cout << "wo : " << odom_rel << endl;
        cout << "last: " << filter_->PostGet()->ExpectedValueGet() << endl;
        // wheel odom cov处理：降维、转类型、做R * cov * R'变换
        {
            // MatrixWrapper::SymmetricMatrix 2dCov;
            // MatrixWrapper::Matrix R(3, 3);
            // R *R.transpose();

            // covFormTrans(odom->pose.covariance, 2dCov);
            // // TODO 还要进行协方差的旋转变换，变换行列数量

            // wo_meas_pdf_->AdditiveNoiseSigmaSet(cov2wraped(downDim(odom->pose.covariance)));

            // cout << cov2wraped(downDim(odom->pose.covariance)) << endl;
        }

        if (predict_test)
        {
            // TODO 一定要确保twist中的速度数据是相对于map世界坐标系
            Vector3 vel_o;
            vel_o.setValue(0.5 * (last_wo_twist_.linear.x + odom->twist.twist.linear.x),
                           0.5 * (last_wo_twist_.linear.y + odom->twist.twist.linear.y),
                           0.5 * (last_wo_twist_.angular.z + odom->twist.twist.angular.z));
            // cout << "true vel: " << vel_o.getX() << ", " << vel_o.getY() << ", " << vel_o.getZ() << endl;
            vel_o *= (odom->header.stamp - last_wo_time_).toSec();

            tf::Vector3 ve = Transform(m2o.rotation_) * vel_o;
            // TODO 获取stamp的变换并进行vector的坐标系转换
            // transformer.transformVector(map_frame, wo_stamp_, Stamped<Vector3>(vel_o, Time(0), odom->header.frame_id), map_frame, ve);
            ColumnVector vel(3);
            vel = 0;
            vel(1) = ve.getX();
            vel(2) = ve.getY();
            vel(3) = ve.getZ();
            // cout << "vel: " << vel << endl;
            filter_->Update(sys_model_, vel, wo_meas_model_, odom_rel);
        }
        else
            filter_->Update(sys_model_, ColumnVector(3) = 0, wo_meas_model_, odom_rel);

        filter_estimate_old_vec_ = filter_->PostGet()->ExpectedValueGet();
        MatrixWrapper::SymmetricMatrix est_cov = filter_->PostGet()->CovarianceGet();
        std::cout << "post: " << filter_estimate_old_vec_ << std::endl;
        tf::Quaternion q;
        q.setRPY(0, 0, filter_estimate_old_vec_(3));
        filter_estimate_old_ = Transform(q, Vector3(filter_estimate_old_vec_(1), filter_estimate_old_vec_(2), 0));

        filter_time_old_ = odom->header.stamp;

        last_wo_time_ = odom->header.stamp;
        last_wo_twist_ = odom->twist.twist;

        // publish tf and topic
        auto m2otf = Transform(m2o.rotation_, m2o.translation_);
        transformer.set_m2o(m2otf, wo_stamp_);
        auto m2orot = Transform(m2o.rotation_, tf::Vector3(0, 0, 0));

        // transformer.set_o2b(m2otf.inverse() * filter_estimate_old_, wo_stamp_);
        transformer.set_o2b(wo_meas_, wo_stamp_);

        transformer.set_m2b_cov(wraped2eigen(est_cov), wo_stamp_);
        // 加入cov

        // TODO 多加spiner，但是不要对wo和lo使用不同的queue，同一queue能保证时间序列的
        cout << "length" << transformer.m2b_cov_list.storage_.size() << endl;

        output_.header.frame_id = map_frame;
        output_.header.stamp = odom->header.stamp;
        // output_.pose.covariance = wraped2cov(filter_->PostGet()->CovarianceGet());
        poseTFToMsg(filter_estimate_old_, output_.pose.pose);
        pose_pub.publish(output_);
        if (broadcastTF)
            pose_broadcaster_.sendTransform(StampedTransform(filter_estimate_old_, odom->header.stamp, map_frame, base_footprint_frame));

        ROS_INFO_STREAM("wo out: " << ros::Time::now() << endl);
        filter_time_old_ = wo_stamp_;
        // TODO 发布融合后打上了stamp的tf、pose topic、cov
    }

}
