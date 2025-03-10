cmake_minimum_required(VERSION 3.8)
project(fast_lio)

#-------------------#
# 기본 빌드 타입 설정
#-------------------#
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

#-------------------#
# C++17로 지정
#-------------------#
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

#---------------------------------------------------------------#
# cmake 폴더에 있는 사용자 정의 Find 모듈(예: Findgpu_bbs3d.cmake) 사용
#---------------------------------------------------------------#
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

#---------------------------------------------#
# 추가 최적화 플래그 (예: -O3, -fexceptions 등)
#---------------------------------------------#
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -pthread -fexceptions")

#-----------------------------------#
# 소스 코드에서 ROOT_DIR로 사용하도록
#-----------------------------------#
add_definitions(-DROOT_DIR=\"${CMAKE_CURRENT_SOURCE_DIR}/\")

set(CMAKE_POSITION_INDEPENDENT_CODE ON)

message("Current CPU architecture: ${CMAKE_SYSTEM_PROCESSOR}")

#----------------------------------#
# CPU 코어 수에 따라 병렬처리 매크로
#----------------------------------#
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

#----------------------#
# OpenMP 사용 (선택적)
#----------------------#
find_package(OpenMP QUIET)
if(OPENMP_FOUND)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS}   ${OpenMP_C_FLAGS}")
endif()

#-------------------------#
# Python (matplotlib 등)
#-------------------------#
find_package(PythonLibs REQUIRED)
find_path(MATPLOTLIB_CPP_INCLUDE_DIRS "matplotlibcpp.h")

#-------------------#
# ROS2 필수 의존성
#-------------------#
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

#---------------------------#
# relocalization_bbs3d (액션)
#---------------------------#
find_package(relocalization_bbs3d REQUIRED)

#-----------------------------------#
# 메시지 빌드(예: Pose6D.msg 사용 시)
#-----------------------------------#
find_package(rosidl_default_generators REQUIRED)
find_package(rosidl_default_runtime REQUIRED)

# 메시지 목록
set(msg_files
  "msg/Pose6D.msg"
)

# 메시지 인터페이스 생성
rosidl_generate_interfaces(${PROJECT_NAME}
  ${msg_files}
)
# 메시지 런타임 의존
ament_export_dependencies(rosidl_default_runtime)

#------------------------------------------------#
# ament_target_dependencies에서 쓸 목록 정리
#------------------------------------------------#
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

#--------------------------------------------#
# Eigen, PCL 등 외부 라이브러리 찾기
#--------------------------------------------#
find_package(Eigen3 REQUIRED)
find_package(PCL REQUIRED COMPONENTS common io)

message(STATUS "Eigen include: ${EIGEN3_INCLUDE_DIR}")
message(STATUS "PCL includes: ${PCL_INCLUDE_DIRS}")

#--------------------------------------------#
# 실행 파일 (fastlio_mapping) 빌드
#--------------------------------------------#
add_executable(fastlio_mapping
  src/laser_mapping.cpp      # 메인 노드
  src/preprocess.cpp         # 전처리 로직
  include/ikd-Tree/ikd_Tree.cpp  # IKD-Tree (있다면)
  # 필요한 .cpp 파일 추가 (예: IMU_Processing.cpp 등)
)

target_include_directories(fastlio_mapping PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
  ${PCL_INCLUDE_DIRS}
  ${PYTHON_INCLUDE_DIRS}
  ${CMAKE_INSTALL_PREFIX}/include
  # matplotlibcpp.h 경로가 필요하면 추가:
  # ${MATPLOTLIB_CPP_INCLUDE_DIRS}
)

target_link_libraries(fastlio_mapping
  ${PCL_LIBRARIES}
  ${PYTHON_LIBRARIES}
  Eigen3::Eigen
  # relocalization_bbs3d 메시지 타입 서포트
  relocalization_bbs3d__rosidl_typesupport_cpp
)

#----------------------#
# 메시지 타입 서포트
#----------------------#
list(APPEND EOL_LIST "foxy" "galactic" "eloquent" "dashing" "crystal")
if($ENV{ROS_DISTRO} IN_LIST EOL_LIST)
  # 예전 ROS2 (eloquent 등)
  rosidl_target_interfaces(fastlio_mapping
    ${PROJECT_NAME} "rosidl_typesupport_cpp")
else()
  # 최근 ROS2 (foxy, galactic 등)
  rosidl_get_typesupport_target(cpp_typesupport_target
    ${PROJECT_NAME} "rosidl_typesupport_cpp")
  target_link_libraries(fastlio_mapping ${cpp_typesupport_target})
endif()

#------------------------------------#
# ament_target_dependencies 등록
#------------------------------------#
ament_target_dependencies(fastlio_mapping
  ${dependencies}
)

#------------------------------------#
# 실행 파일 설치
#------------------------------------#
install(TARGETS fastlio_mapping
  DESTINATION lib/${PROJECT_NAME}
)

#-------------------------------------#
# include/ 등 기타 폴더 설치
#-------------------------------------#
install(
  DIRECTORY include/
  DESTINATION include/
)

# 만약 config/ launch/ rviz/ rviz_cfg/ Log/ PCD/ 등의 폴더도 있으면:
install(
  DIRECTORY config launch rviz rviz_cfg Log PCD
  DESTINATION share/${PROJECT_NAME}
  OPTIONAL
)

#--------------------------#
# ament 패키지 선언
#--------------------------#
ament_package()
