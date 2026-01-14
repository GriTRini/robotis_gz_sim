import rclpy
from rclpy.node import Node
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
import numpy as np
import time

# dsrpy 모듈 import
from robotis_gz_sim.dsrpy import trajectory as traj

class TrajectoryBridge(Node):
    def __init__(self):
        super().__init__('trajectory_bridge')

        # 1. 퍼블리셔 생성
        self.publisher_ = self.create_publisher(
            JointTrajectory,
            '/arm_controller/joint_trajectory',
            10
        )

        # 2. Trajectory Generator 초기화
        self.trajgen = traj.TrajGenerator()
        
        self.dof = 6
        self.joint_names = ['joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6']

        # ---------------------------------------------------------
        # [설정] 입력 데이터
        # ---------------------------------------------------------
        # 1) 1차 목표: 시작 관절 각도 (Radian)
        self.start_angles_rad = np.array([0.7888, 0.6655, 1.5500, 0.04141, 0.90969, -0.8074])
        self.start_angles_deg = np.rad2deg(self.start_angles_rad) # Degree 변환

        # 2) 2차 목표: 최종 Target T-Matrix (4x4)
        self.target_tmat = np.array([
            [ 0.866, 0.0,   0.5,    0.38 ],
            [ 0.3,   0.8,  -0.519, -0.27 ],
            [-0.4,   0.6,   0.693, -0.03 ],
            [ 0.0,   0.0,   0.0,    1.0  ]
        ])

        # ---------------------------------------------------------
        # [상태 머신 변수]
        # 0: 대기 (시작 전)
        # 1: 1차 목표(관절 각도)로 이동 중
        # 2: 1차 목표 도착 -> FK 계산 -> 차이 확인 -> 2차 명령 전송
        # 3: 2차 목표(Target T-Mat)로 이동 중 (종료)
        # ---------------------------------------------------------
        self.mission_state = 0 
        
        # 타이머 설정 (1ms 주기)
        self.timer_period = 0.001 
        self.timer = self.create_timer(self.timer_period, self.timer_callback)

        self.get_logger().info('Trajectory Bridge Node Started.')

    def timer_callback(self):
        # 1. TrajGenerator 업데이트 (시간 적분)
        self.trajgen.update(self.timer_period)

        # 2. 상태 머신(State Machine) 로직 수행
        self.process_mission()

        # 3. ROS 2 메시지 전송 (현재 상태 Publish)
        self.publish_joint_state()

    def process_mission(self):
        """
        단계별 미션 수행 함수
        """
        # [State 0] 시작: 1차 관절 각도로 이동 명령 (trapj)
        if self.mission_state == 0:
            self.get_logger().info(f">>> [Step 1] Moving to Start Angles: {self.start_angles_deg}")
            self.trajgen.trapj(goal_angles=self.start_angles_deg)
            self.mission_state = 1

        # [State 1] 이동 중 확인: 도착했는지 체크
        elif self.mission_state == 1:
            if self.trajgen.goal_reached():
                self.get_logger().info(">>> [Step 1] Arrived at Start Angles.")
                self.mission_state = 2

        # [State 2] FK 계산, 차이 확인, 최종 명령(attrl) 전송
        elif self.mission_state == 2:
            self.calculate_and_move_to_target()
            self.mission_state = 3

        # [State 3] 최종 이동 중 (모니터링용, 추가 동작 없음)
        elif self.mission_state == 3:
            pass 

    def calculate_and_move_to_target(self):
        self.get_logger().info("\n------------------------------------------------")
        self.get_logger().info(">>> [Step 2] Calculating FK and Comparing T-Mats")
        
        # 1. 현재 위치에서 FK 계산
        try:
            current_angles = self.trajgen.angles
            current_tmat = self.trajgen.solve_forward(current_angles)
            
            self.get_logger().info(f"1) Current T-Matrix (Calculated from FK):\n{current_tmat}")
        except AttributeError:
            self.get_logger().error("Error: solve_forward not found.")
            return

        # 2. 목표 T-Matrix 출력
        self.get_logger().info(f"2) Target T-Matrix:\n{self.target_tmat}")

        # 3. 차이(Difference) 확인 (Position 오차 계산)
        # 회전 행렬의 차이는 직관적이지 않으므로 위치(x,y,z) 차이만 계산해서 보여줌
        curr_pos = current_tmat[:3, 3]
        target_pos = self.target_tmat[:3, 3]
        diff_pos = target_pos - curr_pos
        distance = np.linalg.norm(diff_pos)

        self.get_logger().info(f"3) Position Difference (Target - Current):\n   dx, dy, dz = {diff_pos}")
        self.get_logger().info(f"   Euclidean Distance = {distance:.4f} m")
        
        self.get_logger().info("------------------------------------------------")
        self.get_logger().info(">>> [Step 3] Sending [attrl] Command to Target...")

        # 4. attrl 명령 전송
        # kp나 속도를 조절하고 싶으면 인자를 추가하세요 (예: peak_endvel=0.2)
        success = self.trajgen.attrl(goal_tmat=self.target_tmat)

        if success:
            self.get_logger().info(">>> [attrl] Command Accepted. Moving...")
        else:
            self.get_logger().error(">>> [attrl] Command Failed.")

    def publish_joint_state(self):
        """
        현재 TrajGen의 각도를 ROS 메시지로 변환하여 전송
        """
        current_angles_deg = self.trajgen.angles
        current_angles_rad = np.deg2rad(current_angles_deg)

        msg = JointTrajectory()
        msg.joint_names = self.joint_names
        point = JointTrajectoryPoint()
        point.positions = current_angles_rad.tolist()
        point.time_from_start.sec = 0
        point.time_from_start.nanosec = int(self.timer_period * 1e9) 
        msg.points.append(point)
        self.publisher_.publish(msg)

def main(args=None):
    rclpy.init(args=args)
    node = TrajectoryBridge()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()