#ifndef FAST_LIO__LASER_MAPPING_HPP_
#define FAST_LIO__LASER_MAPPING_HPP_

/********************************************************************************************
 * Full LIO-Mapping code, refactored into a Nav2 Lifecycle Node.
 * 
 * Differences from the original:
 *   - Uses nav2_utils::LifecycleNode format (on_configure, on_activate, etc.).
 *   - All map publishing (pointcloud map, path, etc.) has been removed.
 *   - Odometry is published with frame_id = "odom" and child_frame_id = "base_link".
 *   - A lifecycle "reset" (on_deactivate->on_activate or on_cleanup->on_configure)
 *     re-initializes the internal states, effectively restarting the estimator.
 *
 * No code segments omitted to ensure full LIO functionality in this example.
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

#include <Eigen/Core>
#include <pcl/point_types.h>
#include <pcl/filters/voxel_grid.h>

#include "relocalization_bbs3d/action/get_relocalization_pose.hpp"

using PointType = pcl::PointXYZI;

// 전방 선언
struct MeasureGroup;

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
  // 주 타이머 콜백
  void timer_callback();

  // Relocalization 액션 클라이언트 요청/응답
  void request_relocalization();
  void relocalization_response_callback(
    const rclcpp_action::ClientGoalHandle<relocalization_bbs3d::action::GetRelocalizationPose>::WrappedResult &result);

  // publish 유틸
  void publish_cloud_registered();
  void publish_odometry();

  // ROS 멤버
  rclcpp_lifecycle::LifecyclePublisher<nav_msgs::msg::Odometry>::SharedPtr pubOdomAftMapped_;
  rclcpp_lifecycle::LifecyclePublisher<sensor_msgs::msg::PointCloud2>::SharedPtr pubCloudRegistered_{nullptr};

  rclcpp::Subscription<sensor_msgs::msg::Imu>::SharedPtr sub_imu_;
  rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr sub_pcl_pc_;
  rclcpp::Subscription<livox_ros_driver2::msg::CustomMsg>::SharedPtr sub_pcl_livox_;

  std::shared_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster_;
  rclcpp::TimerBase::SharedPtr timer_;

  using GetRelocPose = relocalization_bbs3d::action::GetRelocalizationPose;
  using GetRelocPoseClient = rclcpp_action::Client<GetRelocPose>;
  GetRelocPoseClient::SharedPtr reloc_action_client_;

  // 매개변수 및 상태
  bool path_en_{false};
  bool cloud_registered_en_{false};
  std::string cloud_registered_topic_{""};
  FILE *pos_log_fp_{nullptr};

  std::ofstream fout_pre_, fout_out_, fout_dbg_;
  double epsi_[23];

  pcl::VoxelGrid<PointType> downSizeFilterSurf;
  pcl::VoxelGrid<PointType> downSizeFilterMap;
  int frame_num_{0};
  int kdtree_size_end{0};
};

#endif  // FAST_LIO__LASER_MAPPING_HPP_
