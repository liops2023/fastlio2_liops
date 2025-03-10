#include <omp.h>
#include <math.h>
#include <thread>
#include <fstream>
#include <csignal>
#include <unistd.h>
#include <Python.h>

#include <rclcpp/rclcpp.hpp>
#include <nav2_util/lifecycle_node.hpp>
#include <rclcpp_action/rclcpp_action.hpp>
#include <nav2_util/node_utils.hpp>
#include <tf2_ros/transform_broadcaster.h>

#include "fast_lio/laser_mapping.hpp"

// 아래 헤더들은 실제로 다른 cpp에서 구현된 함수/클래스를 사용하기 위함
#include "fast_lio/common_lib.h"
#include "fast_lio/IMU_Processing.hpp"
#include "fast_lio/preprocess.h"
#include <ikd-Tree/ikd_Tree.h>

#include <Eigen/Core>
#include <Eigen/Eigen>
#include <nav_msgs/msg/odometry.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <sensor_msgs/msg/imu.hpp>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/filters/voxel_grid.h>

#include <relocalization_bbs3d/action/get_relocalization_pose.hpp>

/********************************************************************************************
 * 본 cpp는 질문에 주신 코드(원본 fastLIO mapping node)와 동일하되,
 * - 헤더는 laser_mapping.hpp로 분리
 * - 필요 함수/전역변수는 common_lib, IMU_Processing, preprocess, ikd_Tree 등에 분리
 * - map/path publisher 생략된 버전 (원문 설명)
 ********************************************************************************************/

using namespace std::chrono_literals;

// 전역 (static/anonymous)에 둘 수도 있지만, 여기서는 예시로 namespace internal로 감쌈
namespace internal
{
// === 전역 변수들 ===
static bool flg_exit = false;
static bool flg_first_scan = true;
static bool is_first_lidar = true;
static double last_timestamp_imu = -1.0;
static double last_timestamp_lidar = 0.0;
static double first_lidar_time = 0.0;
static int time_log_counter = 0;
static int scan_count = 0;
static int publish_count = 0;
static bool flg_EKF_inited = false;

// ESEKF, IMU, KD-Tree 관련
static esekfom::esekf<state_ikfom, 12, input_ikfom> kf;
static state_ikfom state_point;
static KD_TREE<PointType> ikdtree;
static MeasureGroup Measures;

// pointcloud 버퍼
static std::deque<double> time_buffer;
static std::deque<PointCloudXYZI::Ptr> lidar_buffer;
static std::deque<sensor_msgs::msg::Imu::ConstSharedPtr> imu_buffer;

// Preprocessor, IMUProcessor
static std::shared_ptr<Preprocess> p_pre(new Preprocess());
static std::shared_ptr<ImuProcess> p_imu(new ImuProcess());

// 다운샘플된 포인트들
static PointCloudXYZI::Ptr feats_undistort(new PointCloudXYZI());
static PointCloudXYZI::Ptr feats_down_body(new PointCloudXYZI());
static PointCloudXYZI::Ptr feats_down_world(new PointCloudXYZI());
static PointCloudXYZI::Ptr normvec(new PointCloudXYZI>(100000, 1));
static PointCloudXYZI::Ptr laserCloudOri(new PointCloudXYZI>(100000, 1));
static PointCloudXYZI::Ptr corr_normvect(new PointCloudXYZI>(100000, 1));

static bool point_selected_surf[MAXN];  // plane selection
static float res_last[100000] = {0.0};

static std::vector<std::vector<int>> pointSearchInd_surf;
static std::vector<PointVector> Nearest_Points;
static std::vector<BoxPointType> cub_needrm;

// 파라미터
static bool scan_pub_en = false;
static bool dense_pub_en = false;
static bool scan_body_pub_en = false;
static bool extrinsic_est_en = true;
static bool runtime_pos_log = false;

static double DET_RANGE = 300.0;
static double time_diff_lidar_to_imu = 0.0;
static bool time_sync_en = false;
static double timediff_lidar_wrt_imu = 0.0;
static bool timediff_set_flg = false;

// Local map bounding box
static BoxPointType LocalMap_Points;
static bool Localmap_Initialized = false;

// 필터 크기
static double filter_size_corner_min = 0.5;
static double filter_size_surf_min   = 0.5;
static double filter_size_map_min    = 0.5;

// fov
static double fov_deg = 180.0;
static double FOV_DEG = 180.0;
static double HALF_FOV_COS = 0.0;

// IMU 공분산
static double gyr_cov = 0.1, acc_cov = 0.1, b_gyr_cov = 0.0001, b_acc_cov = 0.0001;

// 루프 타임
static double match_time = 0, solve_time = 0, solve_const_H_time = 0;
static double kdtree_incremental_time = 0.0, kdtree_search_time = 0.0, kdtree_delete_time = 0.0;

// for debugging
static double T1[MAXN], s_plot[MAXN], s_plot2[MAXN], s_plot3[MAXN];
static double s_plot4[MAXN], s_plot5[MAXN], s_plot6[MAXN], s_plot7[MAXN];
static double s_plot8[MAXN], s_plot9[MAXN], s_plot10[MAXN], s_plot11[MAXN];

// K-F 변동
static int kdtree_size_st = 0, kdtree_delete_counter = 0, add_point_size = 0;

// ekf iteration
static int NUM_MAX_ITERATIONS = 4;

// etc
static int scan_num = 0;
static double lidar_mean_scantime = 0.0;
static bool lidar_pushed = false;
static double lidar_end_time = 0.0;
static int feats_down_size = 0;
static int effct_feat_num = 0;
static double total_residual = 0;
static double res_mean_last = 0.05;

// transform
static nav_msgs::msg::Odometry odomAftMapped;
static geometry_msgs::msg::Quaternion geoQuat;
static V3D euler_cur;
static V3D pos_lid;
static double cube_len = 200.0;

// Mutex
static std::mutex mtx_buffer;
static std::condition_variable sig_buffer;

/** 시그널 핸들러 **/
static void SigHandle(int sig)
{
  flg_exit = true;
  RCLCPP_INFO(rclcpp::get_logger("LIO"), "[SigHandle] Catch SIG %d", sig);
  sig_buffer.notify_all();
  rclcpp::shutdown();
}

// ESEKF 상태 덤프
inline void dump_lio_state_to_log(FILE *fp)
{
  V3D rot_ang = Log(state_point.rot.toRotationMatrix());
  fprintf(fp, "%lf ", Measures.lidar_beg_time - first_lidar_time);
  fprintf(fp, "%lf %lf %lf ", rot_ang(0), rot_ang(1), rot_ang(2));
  fprintf(fp, "%lf %lf %lf ", state_point.pos(0), state_point.pos(1), state_point.pos(2));
  fprintf(fp, "%lf %lf %lf ", 0.0, 0.0, 0.0); // place-holder
  fprintf(fp, "%lf %lf %lf ", state_point.vel(0), state_point.vel(1), state_point.vel(2));
  fprintf(fp, "%lf %lf %lf ", 0.0, 0.0, 0.0); // place-holder
  fprintf(fp, "%lf %lf %lf ", state_point.bg(0), state_point.bg(1), state_point.bg(2));
  fprintf(fp, "%lf %lf %lf ", state_point.ba(0), state_point.ba(1), state_point.ba(2));
  fprintf(fp, "%lf %lf %lf ", state_point.grav[0], state_point.grav[1], state_point.grav[2]);
  fprintf(fp, "\r\n");
  fflush(fp);
}

// Body -> World 좌표 변환 (현재 state_point 사용)
static void pointBodyToWorld(PointType const *const pi, PointType *const po)
{
  V3D p_body(pi->x, pi->y, pi->z);
  V3D p_global = state_point.rot * (state_point.offset_R_L_I * p_body + state_point.offset_T_L_I) + state_point.pos;

  po->x = p_global(0);
  po->y = p_global(1);
  po->z = p_global(2);
  po->intensity = pi->intensity;
}

// 지도에서 제거된 점들 수거
static void points_cache_collect()
{
  PointVector points_history;
  ikdtree.acquire_removed_points(points_history);
}

// local map 이동/리사이즈
static void lasermap_fov_segment()
{
  cub_needrm.clear();
  kdtree_delete_counter = 0;
  kdtree_delete_time = 0.0;

  pos_lid = state_point.pos + state_point.rot * state_point.offset_T_L_I;
  if (!Localmap_Initialized)
  {
    for (int i = 0; i < 3; i++)
    {
      LocalMap_Points.vertex_min[i] = pos_lid(i) - cube_len / 2.0;
      LocalMap_Points.vertex_max[i] = pos_lid(i) + cube_len / 2.0;
    }
    Localmap_Initialized = true;
    RCLCPP_INFO(rclcpp::get_logger("LIO"), "[lasermap_fov_segment] Local map initialized!");
    return;
  }

  float dist_to_map_edge[3][2];
  bool need_move = false;
  for (int i = 0; i < 3; i++)
  {
    dist_to_map_edge[i][0] = fabs(pos_lid(i) - LocalMap_Points.vertex_min[i]);
    dist_to_map_edge[i][1] = fabs(pos_lid(i) - LocalMap_Points.vertex_max[i]);
    if (dist_to_map_edge[i][0] <= MOV_THRESHOLD * DET_RANGE ||
        dist_to_map_edge[i][1] <= MOV_THRESHOLD * DET_RANGE)
    {
      need_move = true;
    }
  }
  if (!need_move)
    return;

  BoxPointType New_LocalMap_Points, tmp_boxpoints;
  New_LocalMap_Points = LocalMap_Points;
  float mov_dist = std::max((cube_len - 2.0 * MOV_THRESHOLD * DET_RANGE) * 0.5 * 0.9,
                            double(DET_RANGE * (MOV_THRESHOLD - 1)));
  for (int i = 0; i < 3; i++)
  {
    tmp_boxpoints = LocalMap_Points;
    if (dist_to_map_edge[i][0] <= MOV_THRESHOLD * DET_RANGE)
    {
      New_LocalMap_Points.vertex_max[i] -= mov_dist;
      New_LocalMap_Points.vertex_min[i] -= mov_dist;
      tmp_boxpoints.vertex_min[i] = LocalMap_Points.vertex_max[i] - mov_dist;
      cub_needrm.push_back(tmp_boxpoints);
    }
    else if (dist_to_map_edge[i][1] <= MOV_THRESHOLD * DET_RANGE)
    {
      New_LocalMap_Points.vertex_max[i] += mov_dist;
      New_LocalMap_Points.vertex_min[i] += mov_dist;
      tmp_boxpoints.vertex_max[i] = LocalMap_Points.vertex_min[i] + mov_dist;
      cub_needrm.push_back(tmp_boxpoints);
    }
  }
  LocalMap_Points = New_LocalMap_Points;

  points_cache_collect();
  double delete_begin = omp_get_wtime();
  if (!cub_needrm.empty())
  {
    kdtree_delete_counter = ikdtree.Delete_Point_Boxes(cub_needrm);
  }
  kdtree_delete_time = omp_get_wtime() - delete_begin;
  RCLCPP_INFO(rclcpp::get_logger("LIO"), "[lasermap_fov_segment] Local map moved. kdtree_delete=%d, time=%.3f",
              kdtree_delete_counter, kdtree_delete_time);
}

// 표준 pointcloud callback
static void standard_pcl_cbk(const sensor_msgs::msg::PointCloud2::SharedPtr msg)
{
  std::unique_lock<std::mutex> lock(mtx_buffer);
  scan_count++;
  double cur_time = get_time_sec(msg->header.stamp);
  double preprocess_start_time = omp_get_wtime();

  if (!is_first_lidar && cur_time < last_timestamp_lidar)
  {
    std::cerr << "[standard_pcl_cbk] Lidar loop back, clear buffer" << std::endl;
    lidar_buffer.clear();
  }
  if (is_first_lidar)
  {
    is_first_lidar = false;
    RCLCPP_INFO(rclcpp::get_logger("LIO"), "[standard_pcl_cbk] First LiDAR scan received!");
  }

  PointCloudXYZI::Ptr ptr(new PointCloudXYZI());
  p_pre->process(msg, ptr);
  lidar_buffer.push_back(ptr);
  time_buffer.push_back(cur_time);
  last_timestamp_lidar = cur_time;
  s_plot11[scan_count] = omp_get_wtime() - preprocess_start_time;

  lock.unlock();
  sig_buffer.notify_all();
}

// Livox 전용 callback
static void livox_pcl_cbk(const livox_ros_driver2::msg::CustomMsg::SharedPtr msg)
{
  std::unique_lock<std::mutex> lock(mtx_buffer);
  double cur_time = get_time_sec(msg->header.stamp);
  double preprocess_start_time = omp_get_wtime();

  scan_count++;
  if (!is_first_lidar && cur_time < last_timestamp_lidar)
  {
    std::cerr << "[livox_pcl_cbk] Lidar loop back, clear buffer" << std::endl;
    lidar_buffer.clear();
  }
  if (is_first_lidar)
  {
    is_first_lidar = false;
    RCLCPP_INFO(rclcpp::get_logger("LIO"), "[livox_pcl_cbk] First LiDAR scan received!");
  }
  last_timestamp_lidar = cur_time;

  if (!time_sync_en && fabs(last_timestamp_imu - last_timestamp_lidar) > 10.0 &&
      !imu_buffer.empty() && !lidar_buffer.empty())
  {
    RCLCPP_WARN(rclcpp::get_logger("LIO"),
                "[livox_pcl_cbk] IMU and LiDAR not synced, IMU=%.6f, LiDAR=%.6f",
                last_timestamp_imu, last_timestamp_lidar);
  }

  if (time_sync_en && !timediff_set_flg && fabs(last_timestamp_lidar - last_timestamp_imu) > 1 && !imu_buffer.empty())
  {
    timediff_set_flg = true;
    timediff_lidar_wrt_imu = last_timestamp_lidar + 0.1 - last_timestamp_imu;
  }

  PointCloudXYZI::Ptr ptr(new PointCloudXYZI());
  p_pre->process(msg, ptr);
  lidar_buffer.push_back(ptr);
  time_buffer.push_back(last_timestamp_lidar);

  s_plot11[scan_count] = omp_get_wtime() - preprocess_start_time;
  lock.unlock();
  sig_buffer.notify_all();
}

// IMU callback
static void imu_cbk(const sensor_msgs::msg::Imu::SharedPtr msg_in)
{
  publish_count++;
  sensor_msgs::msg::Imu::SharedPtr msg(new sensor_msgs::msg::Imu(*msg_in));

  // time offset
  double t_original = get_time_sec(msg_in->header.stamp);
  double t_shifted = t_original - time_diff_lidar_to_imu;
  if (time_sync_en && fabs(timediff_lidar_wrt_imu) > 0.1)
  {
    t_shifted = timediff_lidar_wrt_imu + t_original;
  }

  msg->header.stamp = to_ros_time(t_shifted);

  double timestamp = t_shifted;

  std::unique_lock<std::mutex> lock(mtx_buffer);
  if (timestamp < last_timestamp_imu)
  {
    std::cerr << "[imu_cbk] IMU loop back, clearing buffer" << std::endl;
    imu_buffer.clear();
  }
  last_timestamp_imu = timestamp;
  imu_buffer.push_back(msg);

  lock.unlock();
  sig_buffer.notify_all();
}

// lidar + IMU sync
static bool sync_packages(MeasureGroup &meas)
{
  if (lidar_buffer.empty() || imu_buffer.empty())
  {
    return false;
  }
  if (!lidar_pushed)
  {
    meas.lidar = lidar_buffer.front();
    meas.lidar_beg_time = time_buffer.front();
    if (meas.lidar->points.size() <= 1)
    {
      lidar_end_time = meas.lidar_beg_time + lidar_mean_scantime;
    }
    else if (meas.lidar->points.back().curvature / double(1000) < 0.5 * lidar_mean_scantime)
    {
      lidar_end_time = meas.lidar_beg_time + lidar_mean_scantime;
    }
    else
    {
      scan_num++;
      lidar_end_time = meas.lidar_beg_time + meas.lidar->points.back().curvature / double(1000);
      lidar_mean_scantime +=
        (meas.lidar->points.back().curvature / double(1000) - lidar_mean_scantime) / scan_num;
    }
    meas.lidar_end_time = lidar_end_time;
    lidar_pushed = true;
  }

  if (last_timestamp_imu < lidar_end_time)
  {
    return false;
  }

  meas.imu.clear();
  while (!imu_buffer.empty())
  {
    double imu_time = get_time_sec(imu_buffer.front()->header.stamp);
    if (imu_time > lidar_end_time) break;
    meas.imu.push_back(imu_buffer.front());
    imu_buffer.pop_front();
  }

  lidar_buffer.pop_front();
  time_buffer.pop_front();
  lidar_pushed = false;
  return true;
}

// map에 점 추가
static void map_incremental()
{
  PointVector PointToAdd;
  PointVector PointNoNeedDownsample;
  PointToAdd.reserve(feats_down_size);
  PointNoNeedDownsample.reserve(feats_down_size);

  Nearest_Points.resize(feats_down_size);
  for (int i = 0; i < feats_down_size; i++)
  {
    // body->world
    pointBodyToWorld(&(feats_down_body->points[i]), &(feats_down_world->points[i]));
    if (!Nearest_Points[i].empty() && flg_EKF_inited)
    {
      const PointVector &points_near = Nearest_Points[i];
      bool need_add = true;
      PointType mid_point;
      mid_point.x = floor(feats_down_world->points[i].x / filter_size_map_min) * filter_size_map_min + 0.5 * filter_size_map_min;
      mid_point.y = floor(feats_down_world->points[i].y / filter_size_map_min) * filter_size_map_min + 0.5 * filter_size_map_min;
      mid_point.z = floor(feats_down_world->points[i].z / filter_size_map_min) * filter_size_map_min + 0.5 * filter_size_map_min;
      float dist = calc_dist(feats_down_world->points[i], mid_point);

      if (points_near.size() > 0 &&
          fabs(points_near[0].x - mid_point.x) > 0.5 * filter_size_map_min &&
          fabs(points_near[0].y - mid_point.y) > 0.5 * filter_size_map_min &&
          fabs(points_near[0].z - mid_point.z) > 0.5 * filter_size_map_min)
      {
        PointNoNeedDownsample.push_back(feats_down_world->points[i]);
        continue;
      }

      for (int readd_i = 0; readd_i < NUM_MATCH_POINTS; readd_i++)
      {
        if (points_near.size() < NUM_MATCH_POINTS) break;
        if (calc_dist(points_near[readd_i], mid_point) < dist)
        {
          need_add = false;
          break;
        }
      }
      if (need_add) PointToAdd.push_back(feats_down_world->points[i]);
    }
    else
    {
      PointToAdd.push_back(feats_down_world->points[i]);
    }
  }

  double st_time = omp_get_wtime();
  int added1 = ikdtree.Add_Points(PointToAdd, true);
  int added2 = ikdtree.Add_Points(PointNoNeedDownsample, false);
  add_point_size = added1 + added2;
  kdtree_incremental_time = omp_get_wtime() - st_time;

  RCLCPP_INFO(rclcpp::get_logger("LIO"),
              "[map_incremental] Add %zu, NoDownsample %zu, total added=%d, inc_time=%.5f",
              PointToAdd.size(), PointNoNeedDownsample.size(),
              add_point_size, kdtree_incremental_time);
}

// 측정모델 (h_share_model)
static void h_share_model(state_ikfom &s, esekfom::dyn_share_datastruct<double> &ekfom_data)
{
  double match_start = omp_get_wtime();
  laserCloudOri->clear();
  corr_normvect->clear();
  total_residual = 0.0;

#ifdef MP_EN
  omp_set_num_threads(MP_PROC_NUM);
#pragma omp parallel for
#endif
  for (int i = 0; i < feats_down_size; i++)
  {
    PointType &point_body = feats_down_body->points[i];
    PointType &point_world = feats_down_world->points[i];

    // body->world 변환
    V3D p_body(point_body.x, point_body.y, point_body.z);
    V3D p_global(s.rot * (s.offset_R_L_I * p_body + s.offset_T_L_I) + s.pos);
    point_world.x = p_global(0);
    point_world.y = p_global(1);
    point_world.z = p_global(2);
    point_world.intensity = point_body.intensity;

    std::vector<float> pointSearchSqDis(NUM_MATCH_POINTS);
    auto &points_near = Nearest_Points[i];

    if (ekfom_data.converge)
    {
      ikdtree.Nearest_Search(point_world, NUM_MATCH_POINTS, points_near, pointSearchSqDis);
      if (points_near.size() < NUM_MATCH_POINTS)
        point_selected_surf[i] = false;
      else if (pointSearchSqDis[NUM_MATCH_POINTS - 1] > 5)
        point_selected_surf[i] = false;
      else
        point_selected_surf[i] = true;
    }

    if (!point_selected_surf[i]) continue;

    VF(4) pabcd;
    point_selected_surf[i] = false;
    if (esti_plane(pabcd, points_near, 0.1f))
    {
      float pd2 = pabcd(0) * point_world.x +
                  pabcd(1) * point_world.y +
                  pabcd(2) * point_world.z +
                  pabcd(3);
      float s_scale = 1 - 0.9f * fabs(pd2) / sqrt(p_body.norm());
      if (s_scale > 0.9)
      {
        point_selected_surf[i] = true;
        normvec->points[i].x = pabcd(0);
        normvec->points[i].y = pabcd(1);
        normvec->points[i].z = pabcd(2);
        normvec->points[i].intensity = pd2;
        res_last[i] = fabs(pd2);
      }
    }
  }

  effct_feat_num = 0;
  for (int i = 0; i < feats_down_size; i++)
  {
    if (point_selected_surf[i])
    {
      laserCloudOri->points[effct_feat_num] = feats_down_body->points[i];
      corr_normvect->points[effct_feat_num] = normvec->points[i];
      total_residual += res_last[i];
      effct_feat_num++;
    }
  }

  if (effct_feat_num < 1)
  {
    ekfom_data.valid = false;
    std::cerr << "[h_share_model] No Effective Points!" << std::endl;
    return;
  }

  res_mean_last = total_residual / effct_feat_num;
  match_time += omp_get_wtime() - match_start;
  double solve_start_ = omp_get_wtime();

  ekfom_data.h_x = Eigen::MatrixXd::Zero(effct_feat_num, 12);
  ekfom_data.h.resize(effct_feat_num);

  for (int i = 0; i < effct_feat_num; i++)
  {
    const PointType &laser_p = laserCloudOri->points[i];
    V3D point_this_be(laser_p.x, laser_p.y, laser_p.z);

    M3D point_be_crossmat;
    point_be_crossmat << SKEW_SYM_MATRX(point_this_be);

    V3D point_this = s.offset_R_L_I * point_this_be + s.offset_T_L_I;
    M3D point_crossmat;
    point_crossmat << SKEW_SYM_MATRX(point_this);

    const PointType &norm_p = corr_normvect->points[i];
    V3D norm_vec(norm_p.x, norm_p.y, norm_p.z);

    V3D C = s.rot.conjugate() * norm_vec;
    V3D A = point_crossmat * C;
    if (extrinsic_est_en)
    {
      V3D B = point_be_crossmat * s.offset_R_L_I.conjugate() * C;
      ekfom_data.h_x.block<1, 12>(i, 0) << norm_p.x, norm_p.y, norm_p.z,
          A(0), A(1), A(2),
          B(0), B(1), B(2),
          C(0), C(1), C(2);
    }
    else
    {
      ekfom_data.h_x.block<1, 12>(i, 0) << norm_p.x, norm_p.y, norm_p.z,
          A(0), A(1), A(2),
          0.0, 0.0, 0.0,
          0.0, 0.0, 0.0;
    }
    ekfom_data.h(i) = -norm_p.intensity;
  }
  solve_time += omp_get_wtime() - solve_start_;
}

} // namespace internal

/********************************************************************************************
 * LaserMappingLifecycleNode 구현부
 ********************************************************************************************/

LaserMappingLifecycleNode::LaserMappingLifecycleNode(const rclcpp::NodeOptions &options)
    : nav2_util::LifecycleNode("laser_mapping_lifecycle_node", "", options)
{
  RCLCPP_INFO(this->get_logger(), "Constructing LaserMappingLifecycleNode...");
  // 파라미터 선언
  this->declare_parameter<bool>("publish.path_en", true);
  this->declare_parameter<bool>("publish.scan_publish_en", true);
  this->declare_parameter<bool>("publish.dense_publish_en", true);
  this->declare_parameter<bool>("publish.scan_bodyframe_pub_en", true);
  this->declare_parameter<bool>("publish.cloud_registered_en", false);
  this->declare_parameter<std::string>("publish.cloud_registered_topic", "/cloud_registered");
  this->declare_parameter<int>("max_iteration", 4);
  this->declare_parameter<std::string>("common.lid_topic", "/livox/lidar");
  this->declare_parameter<std::string>("common.imu_topic", "/livox/imu");
  this->declare_parameter<bool>("common.time_sync_en", false);
  this->declare_parameter<double>("common.time_offset_lidar_to_imu", 0.0);
  this->declare_parameter<double>("filter_size_corner", 0.5);
  this->declare_parameter<double>("filter_size_surf", 0.5);
  this->declare_parameter<double>("filter_size_map", 0.5);
  this->declare_parameter<double>("cube_side_length", 200.0);
  this->declare_parameter<float>("mapping.det_range", 300.f);
  this->declare_parameter<double>("mapping.fov_degree", 180.0);
  this->declare_parameter<double>("mapping.gyr_cov", 0.1);
  this->declare_parameter<double>("mapping.acc_cov", 0.1);
  this->declare_parameter<double>("mapping.b_gyr_cov", 0.0001);
  this->declare_parameter<double>("mapping.b_acc_cov", 0.0001);
  this->declare_parameter<double>("preprocess.blind", 0.01);
  this->declare_parameter<int>("preprocess.lidar_type", AVIA);
  this->declare_parameter<int>("preprocess.scan_line", 16);
  this->declare_parameter<int>("preprocess.timestamp_unit", US);
  this->declare_parameter<int>("preprocess.scan_rate", 10);
  this->declare_parameter<int>("point_filter_num", 2);
  this->declare_parameter<bool>("feature_extract_enable", false);
  this->declare_parameter<bool>("runtime_pos_log_enable", false);
  this->declare_parameter<bool>("mapping.extrinsic_est_en", true);
  this->declare_parameter<std::vector<double>>("mapping.extrinsic_T", std::vector<double>());
  this->declare_parameter<std::vector<double>>("mapping.extrinsic_R", std::vector<double>());
}

LaserMappingLifecycleNode::~LaserMappingLifecycleNode()
{
  RCLCPP_INFO(this->get_logger(), "LaserMappingLifecycleNode destructor called.");
}

nav2_util::CallbackReturn
LaserMappingLifecycleNode::on_configure(const rclcpp_lifecycle::State &)
{
  RCLCPP_INFO(get_logger(), "[on_configure]...");

  using namespace internal; // 전역 변수 사용

  // 로드
  path_en_ = this->get_parameter("publish.path_en").as_bool();
  scan_pub_en = this->get_parameter("publish.scan_publish_en").as_bool();
  dense_pub_en = this->get_parameter("publish.dense_publish_en").as_bool();
  scan_body_pub_en = this->get_parameter("publish.scan_bodyframe_pub_en").as_bool();
  cloud_registered_en_ = this->get_parameter("publish.cloud_registered_en").as_bool();
  cloud_registered_topic_ = this->get_parameter("publish.cloud_registered_topic").as_string();
  NUM_MAX_ITERATIONS = this->get_parameter("max_iteration").as_int();

  std::string lid_topic = this->get_parameter("common.lid_topic").as_string();
  std::string imu_topic = this->get_parameter("common.imu_topic").as_string();
  time_sync_en = this->get_parameter("common.time_sync_en").as_bool();
  time_diff_lidar_to_imu = this->get_parameter("common.time_offset_lidar_to_imu").as_double();

  filter_size_corner_min = this->get_parameter("filter_size_corner").as_double();
  filter_size_surf_min   = this->get_parameter("filter_size_surf").as_double();
  filter_size_map_min    = this->get_parameter("filter_size_map").as_double();
  cube_len = this->get_parameter("cube_side_length").as_double();
  DET_RANGE = this->get_parameter("mapping.det_range").as_double();
  fov_deg   = this->get_parameter("mapping.fov_degree").as_double();
  gyr_cov   = this->get_parameter("mapping.gyr_cov").as_double();
  acc_cov   = this->get_parameter("mapping.acc_cov").as_double();
  b_gyr_cov = this->get_parameter("mapping.b_gyr_cov").as_double();
  b_acc_cov = this->get_parameter("mapping.b_acc_cov").as_double();

  p_pre->blind = this->get_parameter("preprocess.blind").as_double();
  p_pre->lidar_type = this->get_parameter("preprocess.lidar_type").as_int();
  p_pre->N_SCANS = this->get_parameter("preprocess.scan_line").as_int();
  p_pre->time_unit = this->get_parameter("preprocess.timestamp_unit").as_int();
  p_pre->SCAN_RATE = this->get_parameter("preprocess.scan_rate").as_int();
  p_pre->point_filter_num = this->get_parameter("point_filter_num").as_int();
  p_pre->feature_enabled  = this->get_parameter("feature_extract_enable").as_bool();
  runtime_pos_log = this->get_parameter("runtime_pos_log_enable").as_bool();
  extrinsic_est_en = this->get_parameter("mapping.extrinsic_est_en").as_bool();

  auto extrinT = this->get_parameter("mapping.extrinsic_T").as_double_array();
  auto extrinR = this->get_parameter("mapping.extrinsic_R").as_double_array();

  // 전역 리셋
  flg_exit = false;
  flg_first_scan = true;
  is_first_lidar = true;
  last_timestamp_imu = -1.0;
  last_timestamp_lidar = 0.0;
  time_log_counter = 0;
  scan_count = 0;
  publish_count = 0;
  flg_EKF_inited = false;

  // fov 관련
  FOV_DEG = (fov_deg + 10.0 > 179.9) ? 179.9 : (fov_deg + 10.0);
  HALF_FOV_COS = cos(FOV_DEG * 0.5 * PI_M / 180.0);

  memset(point_selected_surf, true, sizeof(point_selected_surf));
  memset(res_last, -1000.0f, sizeof(res_last));

  // 다운샘플 초기화
  downSizeFilterSurf.setLeafSize(filter_size_surf_min, filter_size_surf_min, filter_size_surf_min);
  downSizeFilterMap.setLeafSize(filter_size_map_min, filter_size_map_min, filter_size_map_min);

  // IMU 설정
  V3D Lidar_T_wrt_IMU(extrinT[0], extrinT[1], extrinT[2]);
  M3D Lidar_R_wrt_IMU;
  Lidar_R_wrt_IMU << extrinR[0], extrinR[1], extrinR[2],
                     extrinR[3], extrinR[4], extrinR[5],
                     extrinR[6], extrinR[7], extrinR[8];
  p_imu->set_extrinsic(Lidar_T_wrt_IMU, Lidar_R_wrt_IMU);
  p_imu->set_gyr_cov(V3D(gyr_cov, gyr_cov, gyr_cov));
  p_imu->set_acc_cov(V3D(acc_cov, acc_cov, acc_cov));
  p_imu->set_gyr_bias_cov(V3D(b_gyr_cov, b_gyr_cov, b_gyr_cov));
  p_imu->set_acc_bias_cov(V3D(b_acc_cov, b_acc_cov, b_acc_cov));

  // ESEKF init
  for (int i = 0; i < 23; i++)
  {
    epsi_[i] = 0.001;
  }
  kf.init_dyn_share(get_f, df_dx, df_dw, h_share_model, NUM_MAX_ITERATIONS, epsi_);

  // pos_log
  pos_log_fp_ = fopen((std::string(ROOT_DIR) + "/Log/pos_log.txt").c_str(), "w");
  fout_pre_.open(DEBUG_FILE_DIR("mat_pre.txt"), std::ios::out);
  fout_out_.open(DEBUG_FILE_DIR("mat_out.txt"), std::ios::out);
  fout_dbg_.open(DEBUG_FILE_DIR("dbg.txt"), std::ios::out);

  // 퍼블리셔
  pubOdomAftMapped_ = this->create_publisher<nav_msgs::msg::Odometry>(
    "/Odometry", rclcpp::QoS(20));

  tf_broadcaster_ = std::make_shared<tf2_ros::TransformBroadcaster>(this->shared_from_this());

  // 서브스크립션
  if (p_pre->lidar_type == AVIA)
  {
    // livox
    sub_pcl_livox_ = this->create_subscription<livox_ros_driver2::msg::CustomMsg>(
      lid_topic, 20, livox_pcl_cbk);
    RCLCPP_INFO(get_logger(), "[on_configure] Subscribed to Livox: %s, LiDAR type : %d", lid_topic.c_str(), p_pre->lidar_type);
  }
  else
  {
    sub_pcl_pc_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
      lid_topic, rclcpp::SensorDataQoS(), standard_pcl_cbk);
    RCLCPP_INFO(get_logger(), "[on_configure] Subscribed to PC2: %s, LiDAR type : %d", lid_topic.c_str(), p_pre->lidar_type);
  }
  sub_imu_ = this->create_subscription<sensor_msgs::msg::Imu>(
    imu_topic, 10, imu_cbk);

  // relocalization 액션 클라이언트
  reloc_action_client_ = rclcpp_action::create_client<GetRelocPose>(
    this->get_node_base_interface(),
    this->get_node_graph_interface(),
    this->get_node_logging_interface(),
    this->get_node_waitables_interface(),
    "get_relocalization_pose"
  );

  RCLCPP_INFO(get_logger(), "[on_configure] complete. Ready to activate.");
  return nav2_util::CallbackReturn::SUCCESS;
}

nav2_util::CallbackReturn
LaserMappingLifecycleNode::on_activate(const rclcpp_lifecycle::State &)
{
  RCLCPP_INFO(get_logger(), "[on_activate]...");
  pubOdomAftMapped_->on_activate();

  // Reloc action server 체크
  if (!reloc_action_client_->wait_for_action_server(std::chrono::seconds(5))) {
    RCLCPP_ERROR(get_logger(), "Relocalization action server not available after waiting");
  } else {
    RCLCPP_INFO(get_logger(), "Requesting relocalization...");
    request_relocalization();
  }

  if (cloud_registered_en_)
  {
    pubCloudRegistered_ = this->create_publisher<sensor_msgs::msg::PointCloud2>(
      cloud_registered_topic_, rclcpp::QoS(10).reliability(rclcpp::ReliabilityPolicy::Reliable));
    pubCloudRegistered_->on_activate();
    RCLCPP_INFO(get_logger(), "cloud_registered publisher created on topic: %s",
                cloud_registered_topic_.c_str());
  }

  // 타이머
  auto period_ms = std::chrono::milliseconds(static_cast<int64_t>(1000.0 / 100.0)); // 100Hz
  timer_ = this->create_wall_timer(period_ms, std::bind(&LaserMappingLifecycleNode::timer_callback, this));

  RCLCPP_INFO(get_logger(), "[on_activate] done -> ACTIVE state.");
  return nav2_util::CallbackReturn::SUCCESS;
}

nav2_util::CallbackReturn
LaserMappingLifecycleNode::on_deactivate(const rclcpp_lifecycle::State &)
{
  RCLCPP_INFO(get_logger(), "[on_deactivate]...");
  if (timer_)
  {
    timer_->cancel();
    timer_.reset();
  }
  if (pubCloudRegistered_)
    pubCloudRegistered_->on_deactivate();
  pubOdomAftMapped_->on_deactivate();

  RCLCPP_INFO(get_logger(), "[on_deactivate] done -> INACTIVE.");
  return nav2_util::CallbackReturn::SUCCESS;
}

nav2_util::CallbackReturn
LaserMappingLifecycleNode::on_cleanup(const rclcpp_lifecycle::State &)
{
  RCLCPP_INFO(get_logger(), "[on_cleanup]...");

  if (timer_)
  {
    timer_->cancel();
    timer_.reset();
  }
  pubOdomAftMapped_.reset();
  pubCloudRegistered_.reset();
  sub_imu_.reset();
  sub_pcl_pc_.reset();
  sub_pcl_livox_.reset();
  tf_broadcaster_.reset();

  if (pos_log_fp_)
  {
    fclose(pos_log_fp_);
    pos_log_fp_ = nullptr;
  }
  if (fout_pre_.is_open()) fout_pre_.close();
  if (fout_out_.is_open()) fout_out_.close();
  if (fout_dbg_.is_open()) fout_dbg_.close();

  // 전역 리셋
  using namespace internal;
  flg_exit = false;
  flg_first_scan = true;
  is_first_lidar = true;
  last_timestamp_imu = -1.0;
  last_timestamp_lidar = 0.0;
  time_log_counter = 0;
  scan_count = 0;
  publish_count = 0;
  flg_EKF_inited = false;

  RCLCPP_INFO(get_logger(), "[on_cleanup] done.");
  return nav2_util::CallbackReturn::SUCCESS;
}

nav2_util::CallbackReturn
LaserMappingLifecycleNode::on_shutdown(const rclcpp_lifecycle::State &)
{
  RCLCPP_INFO(get_logger(), "[on_shutdown]...");
  if (pos_log_fp_)
  {
    fclose(pos_log_fp_);
    pos_log_fp_ = nullptr;
  }
  if (fout_pre_.is_open()) fout_pre_.close();
  if (fout_out_.is_open()) fout_out_.close();
  if (fout_dbg_.is_open()) fout_dbg_.close();

  return nav2_util::CallbackReturn::SUCCESS;
}

// 주 타이머
void LaserMappingLifecycleNode::timer_callback()
{
  using namespace internal;
  if (sync_packages(Measures))
  {
    if (flg_first_scan)
    {
      first_lidar_time = Measures.lidar_beg_time;
      p_imu->first_lidar_time = first_lidar_time;
      flg_first_scan = false;
      RCLCPP_INFO(get_logger(), "[timer_callback] First scan set. Start alignment!");
      return;
    }

    double t0 = omp_get_wtime();
    match_time = 0;
    kdtree_search_time = 0.0;
    solve_time = 0;
    solve_const_H_time = 0;

    p_imu->Process(Measures, kf, feats_undistort);
    state_point = kf.get_x();
    pos_lid = state_point.pos + state_point.rot * state_point.offset_T_L_I;

    if (feats_undistort->empty())
    {
      RCLCPP_WARN(get_logger(), "No point in current scan, skip");
      return;
    }
    flg_EKF_inited = ((Measures.lidar_beg_time - first_lidar_time) < INIT_TIME) ? false : true;

    lasermap_fov_segment();

    downSizeFilterSurf.setInputCloud(feats_undistort);
    downSizeFilterSurf.filter(*feats_down_body);
    feats_down_size = feats_down_body->points.size();

    if (!ikdtree.Root_Node)
    {
      if (feats_down_size > 5)
      {
        ikdtree.set_downsample_param(filter_size_map_min);
        feats_down_world->resize(feats_down_size);
        for (int i = 0; i < feats_down_size; i++)
        {
          pointBodyToWorld(&(feats_down_body->points[i]), &(feats_down_world->points[i]));
        }
        ikdtree.Build(feats_down_world->points);
        RCLCPP_INFO(this->get_logger(), "[timer_callback] kdtree built with %d pts", feats_down_size);
      }
      return;
    }
    int featsFromMapNum = ikdtree.validnum();
    kdtree_size_st = ikdtree.size();

    if (feats_down_size < 5)
    {
      RCLCPP_WARN(this->get_logger(), "[timer_callback] Too few features, skip this scan");
      return;
    }

    normvec->resize(feats_down_size);
    feats_down_world->resize(feats_down_size);
    pointSearchInd_surf.resize(feats_down_size);
    Nearest_Points.resize(feats_down_size);
    memset(point_selected_surf, true, sizeof(point_selected_surf));

    double t_update_start = omp_get_wtime();
    double solve_H_time = 0;
    kf.update_iterated_dyn_share_modified(LASER_POINT_COV, solve_H_time);
    state_point = kf.get_x();
    euler_cur = SO3ToEuler(state_point.rot);
    pos_lid = state_point.pos + state_point.rot * state_point.offset_T_L_I;

    geoQuat.x = state_point.rot.x();
    geoQuat.y = state_point.rot.y();
    geoQuat.z = state_point.rot.z();
    geoQuat.w = state_point.rot.w();

    double t_update_end = omp_get_wtime();
    map_incremental();
    double t5 = omp_get_wtime();

    if (cloud_registered_en_ && pubCloudRegistered_)
    {
      publish_cloud_registered();
    }

    // publish odom
    publish_odometry();

    double total_time = t5 - t0;
    // runtime log
    if (runtime_pos_log)
    {
      frame_num_++;
      kdtree_size_end = ikdtree.size();
      T1[time_log_counter] = Measures.lidar_beg_time;
      s_plot[time_log_counter]  = t5 - t0;
      s_plot2[time_log_counter] = feats_undistort->points.size();
      s_plot3[time_log_counter] = kdtree_incremental_time;
      s_plot4[time_log_counter] = kdtree_search_time;
      s_plot5[time_log_counter] = kdtree_delete_counter;
      s_plot6[time_log_counter] = kdtree_delete_time;
      s_plot7[time_log_counter] = kdtree_size_st;
      s_plot8[time_log_counter] = kdtree_size_end;
      s_plot10[time_log_counter] = add_point_size;
      s_plot11[time_log_counter] = 0; // temp

      time_log_counter++;

      V3D ext_euler = SO3ToEuler(state_point.offset_R_L_I);
      fout_pre_ << std::setw(20) << (Measures.lidar_beg_time - first_lidar_time)
                << " " << euler_cur.transpose()
                << " " << state_point.pos.transpose()
                << " " << ext_euler.transpose()
                << " " << state_point.offset_T_L_I.transpose()
                << " " << state_point.vel.transpose()
                << " " << state_point.bg.transpose()
                << " " << state_point.ba.transpose()
                << " "
                << state_point.grav[0] << " "
                << state_point.grav[1] << " "
                << state_point.grav[2] << " "
                << feats_undistort->points.size()
                << std::endl;

      dump_lio_state_to_log(pos_log_fp_);

      fout_out_ << std::setw(20) << (Measures.lidar_beg_time - first_lidar_time)
                << " " << euler_cur.transpose()
                << " " << state_point.pos.transpose()
                << " " << ext_euler.transpose()
                << " " << state_point.offset_T_L_I.transpose()
                << " " << state_point.vel.transpose()
                << " " << state_point.bg.transpose()
                << " " << state_point.ba.transpose()
                << " "
                << state_point.grav[0] << " "
                << state_point.grav[1] << " "
                << state_point.grav[2] << " "
                << feats_undistort->points.size()
                << std::endl;
    }
  }
}

void LaserMappingLifecycleNode::publish_cloud_registered()
{
  using namespace internal;
  auto cloud_size = feats_undistort->points.size();
  if (cloud_size == 0) return;

  PointCloudXYZI::Ptr cloud_odom(new PointCloudXYZI());
  cloud_odom->resize(cloud_size);

  for (size_t i = 0; i < cloud_size; i++)
  {
    const auto &pt_in = feats_undistort->points[i];
    auto &pt_out = cloud_odom->points[i];

    // Body->World
    V3D p_body(pt_in.x, pt_in.y, pt_in.z);
    V3D p_global = state_point.rot * (state_point.offset_R_L_I * p_body + state_point.offset_T_L_I) + state_point.pos;

    pt_out.x = p_global(0);
    pt_out.y = p_global(1);
    pt_out.z = p_global(2);
    pt_out.intensity = pt_in.intensity;
  }

  sensor_msgs::msg::PointCloud2 cloud_msg;
  pcl::toROSMsg(*cloud_odom, cloud_msg);
  cloud_msg.header.stamp = to_ros_time(internal::lidar_end_time);
  cloud_msg.header.frame_id = "odom";

  pubCloudRegistered_->publish(cloud_msg);
}

void LaserMappingLifecycleNode::publish_odometry()
{
  using namespace internal;

  odomAftMapped.header.frame_id = "odom";
  odomAftMapped.child_frame_id  = "base_link";
  odomAftMapped.header.stamp    = to_ros_time(lidar_end_time);

  odomAftMapped.pose.pose.position.x = state_point.pos(0);
  odomAftMapped.pose.pose.position.y = state_point.pos(1);
  odomAftMapped.pose.pose.position.z = state_point.pos(2);

  odomAftMapped.pose.pose.orientation.x = geoQuat.x;
  odomAftMapped.pose.pose.orientation.y = geoQuat.y;
  odomAftMapped.pose.pose.orientation.z = geoQuat.z;
  odomAftMapped.pose.pose.orientation.w = geoQuat.w;

  // Cov
  auto P = kf.get_P();
  for (int i = 0; i < 6; i++)
  {
    int k = (i < 3) ? i + 3 : i - 3;
    odomAftMapped.pose.covariance[i * 6 + 0] = P(k, 3);
    odomAftMapped.pose.covariance[i * 6 + 1] = P(k, 4);
    odomAftMapped.pose.covariance[i * 6 + 2] = P(k, 5);
    odomAftMapped.pose.covariance[i * 6 + 3] = P(k, 0);
    odomAftMapped.pose.covariance[i * 6 + 4] = P(k, 1);
    odomAftMapped.pose.covariance[i * 6 + 5] = P(k, 2);
  }

  pubOdomAftMapped_->publish(odomAftMapped);

  geometry_msgs::msg::TransformStamped trans;
  trans.header.frame_id = "odom";
  trans.child_frame_id  = "base_link";
  trans.header.stamp    = odomAftMapped.header.stamp;
  trans.transform.translation.x = odomAftMapped.pose.pose.position.x;
  trans.transform.translation.y = odomAftMapped.pose.pose.position.y;
  trans.transform.translation.z = odomAftMapped.pose.pose.position.z;
  trans.transform.rotation.w = odomAftMapped.pose.pose.orientation.w;
  trans.transform.rotation.x = odomAftMapped.pose.pose.orientation.x;
  trans.transform.rotation.y = odomAftMapped.pose.pose.orientation.y;
  trans.transform.rotation.z = odomAftMapped.pose.pose.orientation.z;

  tf_broadcaster_->sendTransform(trans);
}

void LaserMappingLifecycleNode::request_relocalization()
{
  using Goal = GetRelocPose::Goal;
  auto goal_msg = Goal();
  goal_msg.request = true;

  auto send_goal_options = rclcpp_action::Client<GetRelocPose>::SendGoalOptions();
  send_goal_options.result_callback =
    std::bind(&LaserMappingLifecycleNode::relocalization_response_callback, this, std::placeholders::_1);
  reloc_action_client_->async_send_goal(goal_msg, send_goal_options);
}

void LaserMappingLifecycleNode::relocalization_response_callback(
  const rclcpp_action::ClientGoalHandle<GetRelocPose>::WrappedResult &result)
{
  switch (result.code)
  {
  case rclcpp_action::ResultCode::SUCCEEDED:
    RCLCPP_INFO(get_logger(), "Relocalization succeeded");
    break;
  case rclcpp_action::ResultCode::ABORTED:
    RCLCPP_ERROR(get_logger(), "Relocalization aborted");
    break;
  case rclcpp_action::ResultCode::CANCELED:
    RCLCPP_ERROR(get_logger(), "Relocalization canceled");
    break;
  default:
    RCLCPP_ERROR(get_logger(), "Unknown relocalization result code");
    break;
  }
}

// main 함수
int main(int argc, char **argv)
{
  using namespace internal;
  signal(SIGINT, SigHandle);
  rclcpp::init(argc, argv);

  RCLCPP_INFO(rclcpp::get_logger("LIO"), "[main] Creating LaserMappingLifecycleNode...");
  auto node = std::make_shared<LaserMappingLifecycleNode>();

  rclcpp::executors::MultiThreadedExecutor executor;
  executor.add_node(node->get_node_base_interface());

  RCLCPP_INFO(rclcpp::get_logger("LIO"), "[main] Spinning...");
  executor.spin();

  RCLCPP_INFO(rclcpp::get_logger("LIO"), "[main] Shutdown.");
  rclcpp::shutdown();
  return 0;
}
