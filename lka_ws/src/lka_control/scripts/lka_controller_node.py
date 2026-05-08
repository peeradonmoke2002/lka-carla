#!/usr/bin/python3

import math
import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from carla_msgs.msg import CarlaEgoVehicleControl
from lka_msgs.msg import LaneCenter


class PurePursuitNode(Node):
    def __init__(self):
        super().__init__('pure_pursuit_node')

        self.declare_parameter('wheel_base',        3.0046)
        self.declare_parameter('lane_width',        3.5)
        self.declare_parameter('min_lookahead',     3.0)
        self.declare_parameter('max_lookahead',    10.0)
        self.declare_parameter('ld_velocity_ratio', 2.4)
        self.declare_parameter('max_steer_rad',     1.2217)
        self.declare_parameter('throttle',          0.3)

        self.wheel_base  = self.get_parameter('wheel_base').value
        self.lane_width  = self.get_parameter('lane_width').value
        self.ld_min      = self.get_parameter('min_lookahead').value
        self.ld_max      = self.get_parameter('max_lookahead').value
        self.ld_k        = self.get_parameter('ld_velocity_ratio').value
        self.max_steer   = self.get_parameter('max_steer_rad').value
        self.throttle    = self.get_parameter('throttle').value

        self.center_norm: float | None = None
        self.speed: float = 0.0

        self.create_subscription(LaneCenter, '/lka/lane_center', self.lane_center_callback, 10)
        self.create_subscription(Odometry, '/carla/ego_vehicle/odometry', self.odom_callback, 10)
        self.pub = self.create_publisher(
            CarlaEgoVehicleControl, '/carla/ego_vehicle/vehicle_control_cmd', 10)

        self.create_timer(0.05, self.control_loop)  # 20 Hz

        self.get_logger().info(
            f'Pure Pursuit ready | '
            f'L={self.wheel_base}m  lane={self.lane_width}m  '
            f'ld=[{self.ld_min:.1f}, {self.ld_max:.1f}]m  '
            f'ld_k={self.ld_k}  throttle={self.throttle}'
        )

    # ── Callbacks ──────────────────────────────────────────────────────────

    def lane_center_callback(self, msg: LaneCenter):
        self.center_norm = float(msg.center) if msg.detected else None

    def odom_callback(self, msg: Odometry):
        vx = msg.twist.twist.linear.x
        vy = msg.twist.twist.linear.y
        self.speed = math.sqrt(vx * vx + vy * vy)

    # ── Core Pure Pursuit ──────────────────────────────────────────────────

    def lookahead(self) -> float:
        """Velocity-adaptive lookahead: ld = clamp(ld_k * speed, ld_min, ld_max)."""
        ld = self.ld_k * self.speed
        return float(max(self.ld_min, min(self.ld_max, ld)))

    def compute_steer(self, center_norm: float) -> tuple:
        """
        Pure Pursuit steering from normalized lane center (Autoware-based).

          lateral_error = (center_norm - 0.5) * lane_width   [m, + = right of centre]
          kappa         = 2 * lateral_error / ld²            [curvature, 1/m]
          steer_rad     = atan(wheel_base * kappa)           [steering angle, rad]
          steer_norm    = steer_rad / max_steer              [CARLA: -1..1]

        Returns (steer_norm, ld, lateral_error, steer_rad).
        """
        ld = self.lookahead()
        lateral_error = (center_norm - 0.5) * self.lane_width

        kappa     = 2.0 * lateral_error / (ld ** 2)
        steer_rad = math.atan(self.wheel_base * kappa)
        steer_norm = steer_rad / self.max_steer

        return max(-1.0, min(1.0, steer_norm)), ld, lateral_error, steer_rad

    # ── Control loop ───────────────────────────────────────────────────────

    def control_loop(self):
        cmd = CarlaEgoVehicleControl()
        cmd.throttle          = self.throttle
        cmd.brake             = 0.0
        cmd.hand_brake        = False
        cmd.reverse           = False
        cmd.manual_gear_shift = False

        if self.center_norm is None:
            cmd.throttle = 0.0
            cmd.brake    = 1.0
            cmd.steer    = 0.0
            self.pub.publish(cmd)
            return

        cmd.steer, ld, lat_err, steer_rad = self.compute_steer(self.center_norm)
        self.pub.publish(cmd)

        self.get_logger().info(
            f'speed={self.speed:.1f}m/s  ld={ld:.2f}m  '
            f'lat_err={lat_err:+.3f}m  '
            f'steer={math.degrees(steer_rad):+.1f}°  norm={cmd.steer:+.3f}',
            throttle_duration_sec=1.0,
        )


def main(args=None):
    rclpy.init(args=args)
    node = PurePursuitNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
