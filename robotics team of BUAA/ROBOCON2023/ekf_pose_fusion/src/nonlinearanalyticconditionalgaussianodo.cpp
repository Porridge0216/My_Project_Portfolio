
#include "ekf_pose_fusion/nonlinearanalyticconditionalgaussianodo.h"
#include <bfl/wrappers/rng/rng.h> // Wrapper around several rng libraries
#define NUMCONDARGUMENTS_MOBILE 2

namespace BFL
{
  using namespace MatrixWrapper;

  NonLinearAnalyticConditionalGaussianOdo::NonLinearAnalyticConditionalGaussianOdo(const Gaussian &additiveNoise)
      : AnalyticConditionalGaussianAdditiveNoise(additiveNoise, NUMCONDARGUMENTS_MOBILE),
        df(3, 3)
  {
    // initialize df matrix
    for (unsigned int i = 1; i <= 3; i++)
    {
      for (unsigned int j = 1; j <= 3; j++)
      {
        if (i == j)
          df(i, j) = 1;
        else
          df(i, j) = 0;
      }
    }
  }

  NonLinearAnalyticConditionalGaussianOdo::~NonLinearAnalyticConditionalGaussianOdo() {}

  ColumnVector NonLinearAnalyticConditionalGaussianOdo::ExpectedValueGet() const
  {
    // 只使用3阶状态量来计算
    // state 为3维列向量
    // vel 为3维列向量
    ColumnVector state = ConditionalArgumentGet(0);
    ColumnVector vel = ConditionalArgumentGet(1);
    // 需要知道当前的状态与输入量
    // 猜测：states包含（x y z roll pitch yall）
    //      vel其实不是vel，而是delta(states)，即状态量的微量，要乘以delta(t)时间之后才传进来
    //      且这样默认了为差速轮，不能全向吧

    // state(1) += cos(state(6)) * vel(1);
    // state(2) += sin(state(6)) * vel(1);
    // state(6) += vel(2);

    state(1) += vel(1);
    state(2) += vel(2);
    state(3) += vel(3);

    return state + AdditiveNoiseMuGet();
  }

  Matrix NonLinearAnalyticConditionalGaussianOdo::dfGet(unsigned int i) const
  {
    // 只对status状态参数求导，vel不用导
    // x_dao_k = df * x_dao_(k-1)
    // 同时注意这里i的含义为ConditionalArgumentGet(i)中的i
    // 由于vel不用导，所以i != 0时异常
    if (i == 0) // derivative to the first conditional argument (x)
    {
      // double vel_trans = ConditionalArgumentGet(1)(1);
      // double yaw = ConditionalArgumentGet(0)(6);

      // df(1, 3) = -vel_trans * sin(yaw);
      // df(2, 3) = vel_trans * cos(yaw);

      return df;
    }
    else
    {
      if (i >= NumConditionalArgumentsGet())
      {
        cerr << "This pdf Only has " << NumConditionalArgumentsGet() << " conditional arguments\n";
        exit(-BFL_ERRMISUSE);
      }
      else
      {
        cerr << "The df is not implemented for the" << i << "th conditional argument\n";
        exit(-BFL_ERRMISUSE);
      }
    }
  }

} // namespace BFL
