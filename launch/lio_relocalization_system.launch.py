import os
from ament_index_python.packages import get_package_share_directory

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution

from launch_ros.actions import LifecycleNode, Node, ComposableNodeContainer
from launch_ros.descriptions import ComposableNode

def generate_launch_description():
    # Package paths
    fast_lio_pkg_dir = get_package_share_directory('fast_lio')
    relocalization_pkg_dir = get_package_share_directory('relocalization_bbs3d')
    
    # Config paths
    default_config_path = os.path.join(fast_lio_pkg_dir, 'config')
    default_rviz_config_path = os.path.join(fast_lio_pkg_dir, 'rviz', 'combined.rviz')
    relocalization_config_file = os.path.join(relocalization_pkg_dir, 'config', 'bbs3d_config.yaml')
    
    # Launch configurations
    use_sim_time = LaunchConfiguration('use_sim_time', default='false')
    config_path = LaunchConfiguration('config_path', default=default_config_path)
    config_file = LaunchConfiguration('config_file', default='mid360.yaml')
    rviz_use = LaunchConfiguration('rviz', default='true')
    rviz_cfg = LaunchConfiguration('rviz_cfg', default=default_rviz_config_path)
    
    # Relocalization Node Container
    relocalization_container = ComposableNodeContainer(
        name='relocalization_container',
        namespace='',
        package='rclcpp_components',
        executable='component_container_mt',
        composable_node_descriptions=[
            ComposableNode(
                package='relocalization_bbs3d',
                plugin='relocalization_bbs3d::relocalizationBBS3DNode',
                name='relocalization_bbs3d_node',
                parameters=[relocalization_config_file],
            ),
        ],
        output='screen'
    )
    
    # Fast-LIO Node
    fast_lio_node = LifecycleNode(
        package='fast_lio',
        executable='fastlio_mapping',
        name='laser_mapping_lifecycle_node',
        namespace='',
        parameters=[
            PathJoinSubstitution([config_path, config_file]),
            {'use_sim_time': use_sim_time}
        ],
        output='screen'
    )
    
    # Lifecycle Manager for Relocalization
    relocalization_lifecycle_manager = Node(
        package='nav2_lifecycle_manager',
        executable='lifecycle_manager',
        name='lifecycle_manager_relocalization',
        output='screen',
        parameters=[{
            'autostart': True,
            'bond_timeout': 0.0,
            'node_names': ['relocalization_bbs3d_node'],
            'configure_timeout': 30.0,
            'activate_timeout': 30.0
        }]
    )
    
    # Lifecycle Manager for Fast-LIO
    # Started after relocalization is ready
    fastlio_lifecycle_manager = Node(
        package='nav2_lifecycle_manager',
        executable='lifecycle_manager',
        name='lifecycle_manager_fastlio',
        output='screen',
        parameters=[{
            'autostart': True,
            'bond_timeout': 0.0,
            'node_names': ['laser_mapping_lifecycle_node'],
            'configure_timeout': 20.0,
            'activate_timeout': 20.0
        }]
    )
    
    # RViz Node
    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        arguments=['-d', rviz_cfg],
        condition=IfCondition(rviz_use),
        output='screen'
    )
    
    return LaunchDescription([
        relocalization_container,
        relocalization_lifecycle_manager,
        # 1초 후에 Fast-LIO 시작
        fast_lio_node,
        # 2초 후에 Fast-LIO lifecycle manager 시작
        fastlio_lifecycle_manager,
        rviz_node
    ])