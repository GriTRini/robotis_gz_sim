import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, RegisterEventHandler, SetEnvironmentVariable
from launch.event_handlers import OnProcessExit
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
import xacro

def generate_launch_description():
    package_name = 'robotis_gz_sim'
    
    pkg_share = get_package_share_directory(package_name)
    urdf_file_path = os.path.join(pkg_share, 'urdf', 'omy_f3m', 'omy_f3m.urdf')
    doc = xacro.process_file(urdf_file_path)
    robot_desc = doc.toxml()

    # 1. Gazebo 리소스 경로 설정
    install_dir = os.path.join(pkg_share, '..')
    if 'GZ_SIM_RESOURCE_PATH' in os.environ:
        gz_sim_resource_path = os.environ['GZ_SIM_RESOURCE_PATH'] + ':' + install_dir
    else:
        gz_sim_resource_path = install_dir

    set_gz_sim_resource_path = SetEnvironmentVariable(
        name='GZ_SIM_RESOURCE_PATH', 
        value=gz_sim_resource_path
    )

    # =======================================================================
    # [추가됨] 2. Clock Bridge (Gazebo 시간 -> ROS 2 /clock)
    # 이것이 있어야 ROS 노드들이 시뮬레이션 시간과 동기화됩니다.
    # =======================================================================
    bridge = Node(
        package='ros_gz_bridge',
        executable='parameter_bridge',
        arguments=['/clock@rosgraph_msgs/msg/Clock[gz.msgs.Clock'],
        output='screen'
    )

    # =======================================================================
    # [수정됨] 3. Robot State Publisher에 use_sim_time 설정 추가
    # =======================================================================
    node_robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        output='screen',
        parameters=[{
            'robot_description': robot_desc,
            'use_sim_time': True  # 필수 설정
        }]
    )

    # Gazebo 시뮬레이터 실행
    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            [os.path.join(get_package_share_directory('ros_gz_sim'),
                          'launch', 'gz_sim.launch.py')]),
        launch_arguments={'gz_args': '-r empty.sdf'}.items(),
    )

    # 로봇 스폰
    spawn_entity = Node(
        package='ros_gz_sim',
        executable='create',
        arguments=['-topic', 'robot_description',
                   '-name', 'omy_f3m',
                   '-z', '3.0'], 
        output='screen'
    )

    # 컨트롤러 스포너 (Joint State)
    load_joint_state_broadcaster = Node(
        package="controller_manager",
        executable="spawner",
        arguments=["joint_state_broadcaster"],
        output="screen",
    )

    # 컨트롤러 스포너 (Position Controller)
    load_position_controller = Node(
        package="controller_manager",
        executable="spawner",
        arguments=["arm_controller"],
        output="screen",
    )

    return LaunchDescription([
        set_gz_sim_resource_path,
        bridge,  # 브리지 노드 추가
        node_robot_state_publisher,
        gazebo,
        spawn_entity,
        
        RegisterEventHandler(
            event_handler=OnProcessExit(
                target_action=spawn_entity,
                on_exit=[load_joint_state_broadcaster],
            )
        ),
        RegisterEventHandler(
            event_handler=OnProcessExit(
                target_action=load_joint_state_broadcaster,
                on_exit=[load_position_controller],
            )
        ),
    ])