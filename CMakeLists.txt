cmake_minimum_required(VERSION 3.8)
project(fast_lio)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

### ============= [중요] C++17로 지정 ============= ###
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

### 추가 최적화, 예: -O3, -fexceptions, -pthread 등
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -pthread -fexceptions")

add_definitions(-DROOT_DIR=\"${CMAKE_CURRENT_SOURCE_DIR}/\")

set(CMAKE_POSITION_INDEPENDENT_CODE ON)

message("Current CPU archtecture: ${CMAKE_SYSTEM_PROCESSOR}")

if(CMAKE_SYSTEM_PROCESSOR MATCHES "(x86)|(X86)|(amd64)|(AMD64)")
  include(ProcessorCount)
  ProcessorCount(N)
  message("Processor number:  ${N}")

  if(N GREATER 4)
    add_definitions(-DMP_EN)
    add_definitions(-DMP_PROC_NUM=3)
    message("core for MP: 3")
  elseif(N GREATER 3)
    add_definitions(-DMP_EN)
    add_definitions(-DMP_PROC_NUM=2)
    message("core for MP: 2")
  else()
    add_definitions(-DMP_PROC_NUM=1)
  endif()
else()
  add_definitions(-DMP_PROC_NUM=1)
endif()

find_package(OpenMP QUIET)
if(OPENMP_FOUND)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS}   ${OpenMP_C_FLAGS}")
endif()

### Python (for Py_xxx), matplotlib 필요 시
find_package(PythonLibs REQUIRED)

# ROS dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclcpp_action REQUIRED)
find_package(rclcpp_components REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(std_srvs REQUIRED)
find_package(visualization_msgs REQUIRED)
find_package(pcl_ros REQUIRED)
find_package(pcl_conversions REQUIRED)
find_package(livox_ros_driver2 REQUIRED)
find_package(nav2_util REQUIRED)
find_package(tf2_ros REQUIRED)

# relocalization_bbs3d (액션)
find_package(relocalization_bbs3d REQUIRED)

set(dependencies
  rclcpp
  rclcpp_action
  rclcpp_components
  geometry_msgs
  nav_msgs
  sensor_msgs
  std_msgs
  std_srvs
  visualization_msgs
  pcl_ros
  pcl_conversions
  livox_ros_driver2
  nav2_util
  tf2_ros
  relocalization_bbs3d
)

find_package(Eigen3 REQUIRED)
find_package(PCL REQUIRED COMPONENTS common io)

message(STATUS "Eigen include: ${EIGEN3_INCLUDE_DIR}")
message(STATUS "PCL includes: ${PCL_INCLUDE_DIRS}")

### 빌드할 실행파일
add_executable(fastlio_mapping
  src/laser_mapping.cpp
  src/common_lib.cpp
  src/IMU_Processing.cpp
  src/preprocess.cpp
  src/ikd_Tree.cpp
  # 필요 시 src/time_utils.cpp ...
)

target_include_directories(fastlio_mapping PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
  ${PCL_INCLUDE_DIRS}
  ${PYTHON_INCLUDE_DIRS}
  ${CMAKE_INSTALL_PREFIX}/include
)

target_link_libraries(fastlio_mapping
  ${PCL_LIBRARIES}
  ${PYTHON_LIBRARIES}
  Eigen3::Eigen
)

ament_target_dependencies(fastlio_mapping ${dependencies})

### 설치: 실행파일
install(TARGETS fastlio_mapping
  DESTINATION lib/${PROJECT_NAME}
)

### 설치: 헤더(및 기타 폴더)
install(
  DIRECTORY include/
  DESTINATION include/
)

install(
  DIRECTORY config launch rviz
  DESTINATION share/${PROJECT_NAME}
  OPTIONAL
)

ament_package()
