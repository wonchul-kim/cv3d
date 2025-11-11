#!/usr/bin/env python3
import math
from collections import deque
import numpy as np
import rclpy
from rclpy.node import Node
from std_srvs.srv import Trigger
from aivot_interfaces_v1.msg import RobotState
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point

def quat_to_rot(qx, qy, qz, qw):
    n = math.sqrt(qx*qx + qy*qy + qz*qz + qw*qw)
    if n == 0.0:  # fallback
        return np.eye(3)
    x, y, z, w = qx/n, qy/n, qz/n, qw/n
    return np.array([
        [1 - 2*(y*y + z*z), 2*(x*y - z*w),     2*(x*z + y*w)],
        [2*(x*y + z*w),     1 - 2*(x*x + z*z), 2*(y*z - x*w)],
        [2*(x*z - y*w),     2*(y*z + x*w),     1 - 2*(x*x + y*y)]
    ], dtype=float)

class TcpLiveChecker(Node):
    '''
    -p tx:=0.000793 -p ty:=0.000745 -p tz:=0.171436 
    '''
    def __init__(self):
        super().__init__('tcp_live_checker')
        # Params

        self.declare_parameter('dx', 0.1854)
        self.declare_parameter('dy', 0.4557)
        self.declare_parameter('dz', 0.2918)

        self.declare_parameter('input_topic', '/dsr01/aiv/state/broadcast')
        self.declare_parameter('tx', 0.000793)
        self.declare_parameter('ty', 0.000745)
        self.declare_parameter('tz', 0.171436)
        self.declare_parameter('window', 60)            # 최근 표본 개수
        self.declare_parameter('publish_markers', True)
        self.declare_parameter('marker_topic', '/vgc10_tcp_checker/markers')
        self.declare_parameter('frame_id', 'base')      # RViz용 frame
        self.declare_parameter('log_every_sec', 1.0)

        self.input_topic = self.get_parameter('input_topic').get_parameter_value().string_value
        self.d = np.array([
            self.get_parameter('dx').get_parameter_value().double_value,
            self.get_parameter('dy').get_parameter_value().double_value,
            self.get_parameter('dz').get_parameter_value().double_value
        ], dtype=float)
        self.t = np.array([
            self.get_parameter('tx').get_parameter_value().double_value,
            self.get_parameter('ty').get_parameter_value().double_value,
            self.get_parameter('tz').get_parameter_value().double_value
        ], dtype=float)
        self.window = int(self.get_parameter('window').get_parameter_value().integer_value if \
                          hasattr(self.get_parameter('window').get_parameter_value(), 'integer_value') else \
                          self.get_parameter('window').get_parameter_value().double_value)
        self.publish_markers = self.get_parameter('publish_markers').get_parameter_value().bool_value
        self.marker_topic = self.get_parameter('marker_topic').get_parameter_value().string_value
        self.frame_id = self.get_parameter('frame_id').get_parameter_value().string_value
        self.log_every = self.get_parameter('log_every_sec').get_parameter_value().double_value

        self.buf = deque(maxlen=self.window)
        self.last_log = self.get_clock().now()

        self.sub = self.create_subscription(RobotState, self.input_topic, self.cb_state, 10)
        if self.publish_markers:
            self.pub_marker = self.create_publisher(Marker, self.marker_topic, 1)

        self.srv_reset = self.create_service(Trigger, 'tcp_checker/reset', self.on_reset)

        self.get_logger().info(
            f'Running tcp_live_checker with t = [{self.t[0]:.6f}, {self.t[1]:.6f}, {self.t[2]:.6f}] (m)\n'
            f'  topic={self.input_topic}, window={self.window}, markers={self.publish_markers}'
        )

    def on_reset(self, req, res):
        self.buf.clear()
        res.success = True
        res.message = 'Cleared buffer.'
        return res

    def cb_state(self, msg: RobotState):
        # parent pose (base->currentTCP or flange) from RobotState.tcp_pos  [mm -> m]
        px = float(msg.tcp_pos.position.x) / 1000.0
        py = float(msg.tcp_pos.position.y) / 1000.0
        pz = float(msg.tcp_pos.position.z) / 1000.0
        qx, qy, qz, qw = float(msg.tcp_pos.orientation.x), float(msg.tcp_pos.orientation.y), float(msg.tcp_pos.orientation.z), float(msg.tcp_pos.orientation.w)
        R = quat_to_rot(qx, qy, qz, qw)

        # apply t: p_tool = p_parent + R_parent * t
        p_tool = np.array([px, py, pz]) + R @ self.t
        self.buf.append(p_tool)

        # log periodically
        now = self.get_clock().now()
        if (now - self.last_log).nanoseconds / 1e9 >= self.log_every and len(self.buf) >= 3:
            self.last_log = now
            arr = np.vstack(self.buf)  # Nx3
            ctr = arr.mean(axis=0)
            rms = float(np.sqrt(np.mean(np.sum((arr - ctr)**2, axis=1)))) * 1000.0  # mm
            span = (arr.max(axis=0) - arr.min(axis=0)) * 1000.0  # mm
            self.get_logger().info(
                f"TCP live stats (N={len(self.buf)}): center=({ctr[0]:.4f},{ctr[1]:.4f},{ctr[2]:.4f}) m, "
                f"RMS={rms:.2f} mm, span(mm)=[{span[0]:.2f},{span[1]:.2f},{span[2]:.2f}]"
            )

            # compared to fixed tcp
            self.last_log = now
            ctr = self.buf[-1]
            fixed = self.d
            rms = float(np.sqrt(np.mean(np.sum((fixed - ctr)**2)))) * 1000.0  # mm
            self.get_logger().info(
                f"TCP live stats (N={len(self.buf)}): center=({ctr[0]:.4f},{ctr[1]:.4f},{ctr[2]:.4f}) m, "
                f"fixed=({fixed[0]:.4f},{fixed[1]:.4f},{fixed[2]:.4f}) m, "
                f"RMS2={rms:.2f} mm]"
            )



        # RViz markers
        if self.publish_markers and len(self.buf) >= 1:
            arr = np.vstack(self.buf)
            ctr = arr.mean(axis=0)

            # center sphere
            m1 = Marker()
            m1.header.frame_id = self.frame_id
            m1.header.stamp = self.get_clock().now().to_msg()
            m1.ns = 'vgc10_tcp_checker'
            m1.id = 0
            m1.type = Marker.SPHERE
            m1.action = Marker.ADD
            m1.pose.position.x, m1.pose.position.y, m1.pose.position.z = ctr.tolist()
            m1.pose.orientation.w = 1.0
            m1.scale.x = m1.scale.y = m1.scale.z = 0.01  # 1cm sphere
            m1.color.r, m1.color.g, m1.color.b, m1.color.a = (0.1, 0.6, 1.0, 0.8)

            # point cloud of last N samples
            m2 = Marker()
            m2.header.frame_id = self.frame_id
            m2.header.stamp = m1.header.stamp
            m2.ns = 'vgc10_tcp_checker'
            m2.id = 1
            m2.type = Marker.POINTS
            m2.action = Marker.ADD
            m2.scale.x = 0.002  # 2mm points
            m2.scale.y = 0.002
            m2.color.r, m2.color.g, m2.color.b, m2.color.a = (1.0, 0.2, 0.2, 0.9)
            m2.points = [Point(x=float(p[0]), y=float(p[1]), z=float(p[2])) for p in arr]

            self.pub_marker.publish(m1)
            self.pub_marker.publish(m2)

def main():
    rclpy.init()
    node = TcpLiveChecker()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
