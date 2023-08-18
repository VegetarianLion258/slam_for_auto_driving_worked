/**
 * @file ceres_types.cc
 * @author Frank Zhang (tanhaozhang@connect.polyu.hk)
 * @brief 
 * @version 0.1
 * @date 2023-08-15
 * 
 * @copyright Copyright (c) 2023
 * 
 */

#ifndef SLAM_IN_AUTO_DRIVING_CH4_CERES_TYPE_H_
#define SLAM_IN_AUTO_DRIVING_CH4_CERES_TYPE_H_

#include <glog/logging.h>

#include "common/eigen_types.h"
#include "ceres/ceres.h"
#include "thirdparty/sophus/sophus/so3.hpp"
#include "ch4/imu_preintegration.h"

template <typename T>
void YawPitchRollToRotationMatrix(const T yaw, const T pitch, const T roll, T R[9]) {
    T y = yaw / T(180.0) * T(M_PI);
    T p = pitch / T(180.0) * T(M_PI);
    T r = roll / T(180.0) * T(M_PI);

    R[0] = cos(y) * cos(p);
    R[1] = -sin(y) * cos(r) + cos(y) * sin(p) * sin(r);
    R[2] = sin(y) * sin(r) + cos(y) * sin(p) * cos(r);
    R[3] = sin(y) * cos(p);
    R[4] = cos(y) * cos(r) + sin(y) * sin(p) * sin(r);
    R[5] = -cos(y) * sin(r) + sin(y) * sin(p) * cos(r);
    R[6] = -sin(p);
    R[7] = cos(p) * sin(r);
    R[8] = cos(p) * cos(r);
};

namespace sad
{

namespace ceres_optimization
{
  class PreintegrationCostFunctionCore {
    public:
     PreintegrationCostFunctionCore(std::shared_ptr<sad::IMUPreintegration> imu_preinit, const Eigen::Vector3d gravaty)
         : preinit_(imu_preinit), dt_(imu_preinit->dt_), grav_(gravaty) {}
     template <typename T>
     bool operator()(const T* const i, const T* const j, T* residual) const {
        Eigen::Matrix<T, 3, 1> r_i(i[0], i[1], i[2]);
        Eigen::Matrix<T, 3, 1> r_j(j[0], j[1], j[2]);
        Eigen::Matrix<T, 3, 1> p_i(i[3], i[4], i[5]);
        Eigen::Matrix<T, 3, 1> p_j(j[3], j[4], j[5]);
        Eigen::Matrix<T, 3, 1> v_i(i[6], i[7], i[8]);
        Eigen::Matrix<T, 3, 1> v_j(j[6], j[7], j[8]);
        Eigen::Matrix<T, 3, 1> bg(i[9], i[10], i[11]);
        Eigen::Matrix<T, 3, 1> ba(i[12], i[13], i[14]);

        Sophus::SO3<double> dR = preinit_->GetDeltaRotation(preinit_->bg_);

        Eigen::Matrix<double, 3, 1> dvd = preinit_->GetDeltaVelocity(preinit_->bg_, preinit_->ba_);
        Eigen::Matrix<T, 3, 1> dv(T(dvd.x()), T(dvd.y()), T(dvd.z()));
        Eigen::Matrix<double, 3, 1> dpd = preinit_->GetDeltaPosition(preinit_->bg_, preinit_->ba_);
        Eigen::Matrix<T, 3, 1> dp(T(dpd.x()), T(dpd.y()), T(dpd.z()));
      
        Sophus::SO3<T, 0> R_i = Sophus::SO3<T, 0>::exp(r_i);
        Sophus::SO3<T, 0> R_j = Sophus::SO3<T, 0>::exp(r_j);

        Eigen::Matrix<T, 3, 1> grav(T(grav_.x()), T(grav_.y()), T(grav_.z()));

        Eigen::Matrix<T, 3, 1> er = (dR.inverse() * R_i.inverse() * R_j).log();
        Eigen::Matrix<T, 3, 3> RiT = R_i.matrix();
        Eigen::Matrix<T, 3, 1> ev = RiT * (v_j - v_i - grav * T(dt_)) - dv;
        Eigen::Matrix<T, 3, 1> ep = RiT * (p_j - p_i - v_i * T(dt_) - grav * T(dt_) * T(dt_) * T(0.5)) - dp;
        residual[0] = T(er[0]);
        residual[1] = T(er[1]);
        residual[2] = T(er[2]);
        residual[3] = T(ev[0]);
        residual[4] = T(ev[1]);
        residual[5] = T(ev[2]);
        residual[6] = T(ep[0]);
        residual[7] = T(ep[1]);
        residual[8] = T(ep[2]);
        return true;
    }

    private:
    const double dt_;
    std::shared_ptr<sad::IMUPreintegration> preinit_ = nullptr;
    const Eigen::Vector3d grav_;
  };

  ceres::CostFunction* CreatePreintegrationCostFunction(std::shared_ptr<sad::IMUPreintegration> imu_preinit, const Eigen::Vector3d gravaty) {
    return new ceres::AutoDiffCostFunction<PreintegrationCostFunctionCore, 9, 15, 15>(new PreintegrationCostFunctionCore(imu_preinit, gravaty));
  }

  class BiasCostFunctionCore {
   public:
    BiasCostFunctionCore(){}
    template <typename T>
    bool operator() (const T* const i, const T* const j, T* residual) const
    {
      Eigen::Matrix<T, 3, 1> bg_i(i[9], i[10], i[11]);
      Eigen::Matrix<T, 3, 1> bg_j(j[9], j[10], j[11]);
      Eigen::Matrix<T, 3, 1> ba_i(i[12], i[13], i[14]);
      Eigen::Matrix<T, 3, 1> ba_j(j[12], j[13], j[14]);
      Eigen::Matrix<T, 3, 1> d_ba = ba_j - ba_i;
      Eigen::Matrix<T, 3, 1> d_bg = bg_j - bg_i;
      residual[0] = T(d_ba[0]);
      residual[1] = T(d_ba[1]);
      residual[2] = T(d_ba[2]);
      residual[3] = T(d_bg[0]);
      residual[4] = T(d_bg[1]);
      residual[5] = T(d_bg[2]);

      return true;
    }
  };
  ceres::CostFunction* CreateBiasConstFunction() {
    return new ceres::AutoDiffCostFunction<BiasCostFunctionCore, 6, 15, 15>(
      new BiasCostFunctionCore()
    );
  }

  class PriorCostFunctionCore {
    public:
     PriorCostFunctionCore(const std::shared_ptr<sad::NavStated> prior) : prior_(prior) {}
     template <typename T>
     bool operator()(const T* const i, T* residual) const {
      Eigen::Vector3d prior_r_d = prior_->R_.log();
      Eigen::Vector3d prior_p_d = prior_->p_;
      Eigen::Vector3d prior_v_d = prior_->v_;
      Eigen::Vector3d prior_bg_d = prior_->bg_;
      Eigen::Vector3d prior_ba_d = prior_->ba_;
      Eigen::Matrix<double, 15, 1> prior_M;
      prior_M << prior_r_d, prior_p_d, prior_v_d, prior_bg_d, prior_ba_d;
      for (int temp = 0; temp < prior_M.size(); temp++)
      {
        residual[temp] = T(prior_M[temp]) - i[temp];
      }
      return true;
    }
    private:
     const std::shared_ptr<sad::NavStated> prior_;
  };
  ceres::CostFunction* CreatePriorCostFunction(const std::shared_ptr<sad::NavStated> prior) {
     return new ceres::AutoDiffCostFunction<PriorCostFunctionCore, 15, 15>(new PriorCostFunctionCore(prior));
  }

  class GnssCostFunctionCore {
    public:
    GnssCostFunctionCore(const Sophus::SE3d gnss_states) : gnss_states_(gnss_states){}
    template <typename T>
    bool operator() (const T* const i, T* residual) const
    {
      Eigen::Matrix<T, 3, 1> r_i(i[0], i[1], i[2]);
      Sophus::SO3<T, 0> R_i = Sophus::SO3<T, 0>::exp(r_i);
      Eigen::Matrix<T, 3, 1> t_i(i[3], i[4], i[5]);
      Eigen::Matrix<T, 3, 1> e_r = (gnss_states_.so3().inverse() * R_i).log();
      Eigen::Matrix<T, 3, 1> e_t = t_i - gnss_states_.translation();
      residual[0] = T(e_r[0]);
      residual[1] = T(e_r[1]);
      residual[2] = T(e_r[2]);
      residual[3] = T(e_t[0]);
      residual[4] = T(e_t[1]);
      residual[5] = T(e_t[2]);
      return true;
    }
    private:
    const Sophus::SE3d gnss_states_;
  };
  static ceres::CostFunction* CreateGnssCostFunction(const Sophus::SE3d gnss_states){
    return new ceres::AutoDiffCostFunction<GnssCostFunctionCore, 6, 15> (
      new GnssCostFunctionCore(gnss_states)
    );
  }

} // namespace ceres_optimization


} //namespace sad

#endif