# file: lio_relocalization_system_no_nav2lm.launch.py

import os

from launch import LaunchDescription
from launch.actions import RegisterEventHandler, EmitEvent, TimerAction
from launch_ros.actions import LifecycleNode
from launch_ros.event_handlers import OnStateTransition
from launch_ros.events.lifecycle import ChangeState
from lifecycle_msgs.msg import Transition as LifecycleTransition

# ament_index_python 을 통해 패키지별 share 디렉토리 경로를 찾을 수 있음
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():

    # 1) 각 패키지의 share 디렉토리 경로를 얻는다
    fast_lio_share_dir = get_package_share_directory('fast_lio')
    relocal_bbs3d_share_dir = get_package_share_directory('relocalization_bbs3d')

    # 2) config 폴더 안의 yaml 파일 경로를 합친다
    fast_lio_config = os.path.join(fast_lio_share_dir, 'config', 'mid360.yaml')
    relocalization_config = os.path.join(relocal_bbs3d_share_dir, 'config', 'bbs3d_config.yaml')

    # [A] Relocalization Lifecycle Node
    relocalization_node = LifecycleNode(
        package='relocalization_bbs3d',
        executable='relocalization_bbs3d_node',
        name='relocalization_bbs3d_node',
        namespace='',
        output='screen',
        parameters=[relocalization_config],  # YAML 파일 지정
    )

    # [B] Fast-LIO Lifecycle Node
    fast_lio_node = LifecycleNode(
        package='fast_lio',
        executable='fastlio_mapping',
        name='laser_mapping_lifecycle_node',
        namespace='',
        output='screen',
        parameters=[fast_lio_config],        # YAML 파일 지정
    )

    # ---(C) Reloc 노드를 일정 시간 뒤에 configure -> activate--- #
    relocalization_configure_evt = TimerAction(
        period=2.0,  # 2초 뒤에 Configure
        actions=[
            EmitEvent(event=ChangeState(
                lifecycle_node_matcher=lambda node: node if node == relocalization_node else None,
                transition_id=LifecycleTransition.TRANSITION_CONFIGURE
            ))
        ]
    )
    relocalization_activate_evt = TimerAction(
        period=4.0,  # 4초 뒤에 Activate
        actions=[
            EmitEvent(event=ChangeState(
                lifecycle_node_matcher=lambda node: node if node == relocalization_node else None,
                transition_id=LifecycleTransition.TRANSITION_ACTIVATE
            ))
        ]
    )

    # ---(D) Reloc 노드가 active가 되면 Fast-LIO를 configure -> activate--- #
    event_handler = RegisterEventHandler(
        OnStateTransition(
            target_lifecycle_node=relocalization_node,
            goal_state='active',
            entities=[
                EmitEvent(event=ChangeState(
                    lifecycle_node_matcher=lambda node: node if node == fast_lio_node else None,
                    transition_id=LifecycleTransition.TRANSITION_CONFIGURE
                )),
                EmitEvent(event=ChangeState(
                    lifecycle_node_matcher=lambda node: node if node == fast_lio_node else None,
                    transition_id=LifecycleTransition.TRANSITION_ACTIVATE
                ))
            ]
        )
    )

    return LaunchDescription([
        # Lifecycle Node 등록
        relocalization_node,
        fast_lio_node,

        # Reloc autostart (configure->activate)
        relocalization_configure_evt,
        relocalization_activate_evt,

        # Reloc active -> Fast-LIO configure->activate
        event_handler
    ])
