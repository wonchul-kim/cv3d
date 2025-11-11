#!/usr/bin/env python3
'''
# 3) 각 포즈마다(정지 후) 저장
ros2 service call /save_pose std_srvs/srv/Trigger "{}"
'''
import os, csv, math
import rclpy
from rclpy.node import Node
from rclpy.time import Time
from rclpy.duration import Duration
from std_srvs.srv import Trigger
from aivot_interfaces_v1.msg import RobotState

def _norm(qx, qy, qz, qw):
    return math.sqrt(qx*qx + qy*qy + qz*qz + qw*qw)

class PivotPoseRecorderFromRobotState(Node):
    def __init__(self):
        super().__init__('pivot_pose_recorder_from_robotstate')

        # Parameters
        self.declare_parameter('input_topic', '/dsr01/aiv/state/broadcast')
        self.declare_parameter('output_csv', '/wonchul/outputs/pivot_cal/pivot_poses.csv')
        self.declare_parameter('require_still', True)   # moving==False일 때만 저장
        self.declare_parameter('freshness_sec', 0.5)    # 최신 메시지 한정
        self.declare_parameter('parent_frame_label', 'flange_or_current_tcp')
        # ↑ CSV/로그에 표기용 라벨(실제 TF는 솔버에서 생성)

        self.input_topic = self.get_parameter('input_topic').get_parameter_value().string_value
        self.output_csv  = self.get_parameter('output_csv').get_parameter_value().string_value
        self.require_still = self.get_parameter('require_still').get_parameter_value().bool_value
        self.freshness = self.get_parameter('freshness_sec').get_parameter_value().double_value
        self.parent_label = self.get_parameter('parent_frame_label').get_parameter_value().string_value

        # State buffer
        self._last_msg = None
        self._last_msg_time = None

        # Subscriptions / Service
        self.sub = self.create_subscription(RobotState, self.input_topic, self.cb_state, 10)
        self.srv = self.create_service(Trigger, 'save_pose', self.handle_save_pose)

        # CSV 준비
        os.makedirs(os.path.dirname(self.output_csv), exist_ok=True)
        self.header_written = os.path.exists(self.output_csv) and os.path.getsize(self.output_csv) > 0
        self.saved_count = 0

        self.get_logger().info(
            f'PivotPoseRecorderFromRobotState ready.\n'
            f'  input_topic  = {self.input_topic}\n'
            f'  output_csv   = {self.output_csv}\n'
            f'  require_still= {self.require_still}\n'
            f'  freshness(s) = {self.freshness}\n'
            f'  parent_label = {self.parent_label}\n'
            f'사용법: 각 포즈마다 정지 후 아래 명령으로 저장\n'
            f'  ros2 service call /save_pose std_srvs/srv/Trigger "{{}}"\n'
            f'주의: tcp_pos.position은 Doosan 기준 mm → m로 변환됩니다.'
        )

    def cb_state(self, msg: RobotState):
        self._last_msg = msg
        # RobotState.header.stamp 가 들어온다면 사용, 없으면 rclpy now
        try:
            self._last_msg_time = Time.from_msg(msg.header.stamp)
        except Exception:
            self._last_msg_time = self.get_clock().now()

    def handle_save_pose(self, request, response):
        if self._last_msg is None:
            response.success = False
            response.message = '아직 RobotState 메시지를 받지 못했습니다.'
            return response

        # 최신성 체크
        now = self.get_clock().now()
        if self._last_msg_time is not None:
            age = (now - self._last_msg_time).nanoseconds / 1e9
            if age > self.freshness:
                response.success = False
                response.message = f'RobotState가 오래되었습니다(age={age:.2f}s). 로봇이 움직이는 중이거나 퍼블리셔 지연일 수 있습니다.'
                return response

        # 정지 조건(옵션)
        if self.require_still and getattr(self._last_msg, 'moving', False):
            response.success = False
            response.message = '로봇이 moving=True 입니다. 정지 후 저장하세요(require_still=true).'
            return response

        # Pose 추출 (mm → m)
        pos = self._last_msg.tcp_pos.position
        ori = self._last_msg.tcp_pos.orientation

        x_m = float(pos.x) / 1000.0
        y_m = float(pos.y) / 1000.0
        z_m = float(pos.z) / 1000.0

        qx, qy, qz, qw = float(ori.x), float(ori.y), float(ori.z), float(ori.w)
        n = _norm(qx, qy, qz, qw)
        if n == 0.0:
            response.success = False
            response.message = 'Quaternion 노름이 0입니다. 유효한 자세가 아닙니다.'
            return response
        # 정규화(안전)
        qx, qy, qz, qw = qx/n, qy/n, qz/n, qw/n

        # CSV 기록
        write_header = not self.header_written
        with open(self.output_csv, 'a', newline='') as f:
            w = csv.writer(f)
            if write_header:
                w.writerow(['x','y','z','qx','qy','qz','qw','parent_label','status','moving'])
                self.header_written = True
            w.writerow([x_m, y_m, z_m, qx, qy, qz, qw, self.parent_label,
                        int(self._last_msg.status), int(self._last_msg.moving)])
            self.saved_count += 1

        response.success = True
        response.message = (f'Saved #{self.saved_count}: '
                            f'[{x_m:.6f}, {y_m:.6f}, {z_m:.6f}, {qx:.6f}, {qy:.6f}, {qz:.6f}, {qw:.6f}] '
                            f'(parent={self.parent_label})')
        self.get_logger().info(response.message)
        return response

def main():
    rclpy.init()
    node = PivotPoseRecorderFromRobotState()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
