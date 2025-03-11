#ifndef FAST_LIO__LASER_MAPPING_HPP_
#define FAST_LIO__LASER_MAPPING_HPP_

/********************************************************************************************
 * Full LIO-Mapping code, refactored into a Nav2 Lifecycle Node.
 *
 * Differences from the original:
 *   - Uses nav2_utils::LifecycleNode format (on_configure, on_activate, etc.).
 *   - Map/Path publishing omitted (we only do odom, tf, optional "cloud_registered").
 *   - Odometry is published with frame_id = "odom" and child_frame_id = "base_link".
 *   - Lifecycle "reset" (on_deactivate->on_activate or on_cleanup->on_configure)
 *     re-initializes internal states, effectively restarting the estimator.
 ********************************************************************************************/

#include <chrono>
#include <memory>
#include <string>
#include <vector>
#include <fstream>

#include <nav2_util/lifecycle_node.hpp>
#include <rclcpp_action/rclcpp_action.hpp>
#include <rclcpp_lifecycle/lifecycle_publisher.hpp>
#include <tf2_ros/transform_broadcaster.h>

#include <nav_msgs/msg/odometry.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <sensor_msgs/msg/imu.hpp>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <livox_ros_driver2/msg/custom_msg.hpp>

#include <pcl/filters/voxel_grid.h>

// 원본 액션 (Reloc)
#include <relocalization_bbs3d/action/get_relocalization_pose.hpp>

/**
 * common_lib.h 내부에서 이미 다음을 포함한다고 가정:
 *   typedef pcl::PointXYZINormal PointType;
 *   + 여러 유틸 매크로/함수 (esti_plane, time_list, ...)
 */
#include "fast_lio/common_lib.h"  // <-- 여기서 PointType, etc. 정의됨
#include "fast_lio/IMU_Processing.hpp"
#include "fast_lio/preprocess.h"
#include <ikd-Tree/ikd_Tree.h>  // ikdtree

//---------------------------------------------------------------------
// LaserMappingLifecycleNode
//---------------------------------------------------------------------
class LaserMappingLifecycleNode : public nav2_util::LifecycleNode
{
public:
  explicit LaserMappingLifecycleNode(const rclcpp::NodeOptions &options = rclcpp::NodeOptions());
  ~LaserMappingLifecycleNode() override;

protected:
  // nav2 라이프사이클 콜백
  nav2_util::CallbackReturn on_configure(const rclcpp_lifecycle::State &) override;
  nav2_util::CallbackReturn on_activate(const rclcpp_lifecycle::State &) override;
  nav2_util::CallbackReturn on_deactivate(const rclcpp_lifecycle::State &) override;
  nav2_util::CallbackReturn on_cleanup(const rclcpp_lifecycle::State &) override;
  nav2_util::CallbackReturn on_shutdown(const rclcpp_lifecycle::State &) override;

private:
  // 주 타이머 콜백 (메인 loop)
  void timer_callback();

  // publish 유틸
  void publish_cloud_registered();
  void publish_odometry();

  // Relocalization 액션 클라이언트
  void request_relocalization();
  void relocalization_response_callback(
    const rclcpp_action::ClientGoalHandle<relocalization_bbs3d::action::GetRelocalizationPose>::WrappedResult &result);
  
  // map->odom 변환 정보를 PoseStamped 메시지 형태로 구독하는 콜백 함수
  void map_to_odom_pose_cb(const geometry_msgs::msg::PoseStamped::SharedPtr msg);

  // ROS 멤버
  rclcpp_lifecycle::LifecyclePublisher<nav_msgs::msg::Odometry>::SharedPtr pubOdomAftMapped_;
  rclcpp_lifecycle::LifecyclePublisher<sensor_msgs::msg::PointCloud2>::SharedPtr pubCloudRegistered_{nullptr};

  rclcpp::Subscription<sensor_msgs::msg::Imu>::SharedPtr sub_imu_;
  rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr sub_pcl_pc_;
  rclcpp::Subscription<livox_ros_driver2::msg::CustomMsg>::SharedPtr sub_pcl_livox_;

  std::shared_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster_;
  rclcpp::TimerBase::SharedPtr timer_;

  // Reloc action
  using GetRelocPose = relocalization_bbs3d::action::GetRelocalizationPose;
  using GetRelocPoseClient = rclcpp_action::Client<GetRelocPose>;
  GetRelocPoseClient::SharedPtr reloc_action_client_;

  // 노드 파라미터/상태
  bool path_en_{false};
  bool cloud_registered_en_{false};
  std::string cloud_registered_topic_{""};
  FILE *pos_log_fp_{nullptr};

  // Debug 출력 파일
  std::ofstream fout_pre_, fout_out_, fout_dbg_;
  double epsi_[23] = {0.0};

  // 다운샘플 필터
  pcl::VoxelGrid<PointType> downSizeFilterSurf;
  pcl::VoxelGrid<PointType> downSizeFilterMap;

  // 내부 상태
  int frame_num_{0};
  int kdtree_size_end{0};

  //=== 재시도용 스레드 관련 ===
  std::thread reloc_attempt_thread_;        // 재시도 스레드
  std::atomic<bool> reloc_stop_flag_{false}; // stop 여부
  double reloc_score_threshold_ = 100.0;     // 점수 임계치
  int reloc_timeout_sec_ = 300;             // 최대 재시도 시간(초)
  bool enable_relocalization_ = true;

  void start_reloc_retry_loop(); // 재시도 루프 시작 함수
  void reloc_retry_loop();       // 스레드 내에서 동작할 실제 루프

  std::string map_frame_id_;
  std::string odom_frame_id_;
  std::string sensor_frame_id_;
  
  rclcpp::Subscription<geometry_msgs::msg::PoseStamped>::SharedPtr sub_map_to_odom_pose_;
  std::mutex map_to_odom_mtx_;
  geometry_msgs::msg::Transform map_to_odom_; // Relocalization 으로 획득한 값 저장
  bool have_map_to_odom_{false}; // map -> odom 값을 갖고 있는 지 여부
  
};

#endif  // FAST_LIO__LASER_MAPPING_HPP_
