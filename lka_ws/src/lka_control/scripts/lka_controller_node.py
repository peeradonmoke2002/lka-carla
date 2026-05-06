#!/usr/bin/env python3
"""
Pure Pursuit LKA Controller

Adapted from Autoware autoware_pure_pursuit:
  - Lookahead distance: ld = clamp(ld_vel_ratio * speed, min_ld, max_ld)
  - Lateral error:      e  = (center_norm - 0.5) * lane_width  [metres]
  - Curvature:          κ  = 2 * e / ld²
  - Steering angle:     δ  = atan(wheel_base * κ)              [rad]
  - CARLA steer:        δ / max_steer_rad  → [-1, 1]

Subscribed topics:
  /lka/lane_center                 std_msgs/Float32  – normalised x [0,1]; -1 = no detection
  /carla/ego_vehicle/odometry      nav_msgs/Odometry – for current speed

Published topic:
  /carla/ego_vehicle/vehicle_control_cmd  carla_msgs/CarlaEgoVehicleControl

Parameters:
  wheel_base        (float) – wheelbase [m]            default 2.875  (CARLA Lincoln MKZ)
  lane_width        (float) – lane width [m]            default 3.5
  min_lookahead     (float) – min lookahead distance    default 3.0 m
  max_lookahead     (float) – max lookahead distance    default 10.0 m
  ld_velocity_ratio (float) – speed → lookahead ratio   default 2.4   (from Autoware)
  max_steer_rad     (float) – vehicle max steer angle   default 1.22 rad (~70°)
  throttle          (float) – constant throttle [0,1]   default 0.3
"""

import math
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32
from nav_msgs.msg import Odometry
from carla_msgs.msg import CarlaEgoVehicleControl


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

        self._L          = self.get_parameter('wheel_base').get_parameter_value().double_value
        self._lw         = self.get_parameter('lane_width').get_parameter_value().double_value
        self._ld_min     = self.get_parameter('min_lookahead').get_parameter_value().double_value
        self._ld_max     = self.get_parameter('max_lookahead').get_parameter_value().double_value
        self._ld_k       = self.get_parameter('ld_velocity_ratio').get_parameter_value().double_value
        self._max_steer  = self.get_parameter('max_steer_rad').get_parameter_value().double_value
        self._throttle   = self.get_parameter('throttle').get_parameter_value().double_value

        self._center_norm: float | None = None
        self._speed: float = 0.0

        self.create_subscription(Float32,  '/lka/lane_center',             self._cb_center, 10)
        self.create_subscription(Odometry, '/carla/ego_vehicle/odometry',  self._cb_odom,   10)
        self.pub = self.create_publisher(
            CarlaEgoVehicleControl, '/carla/ego_vehicle/vehicle_control_cmd', 10)

        self.create_timer(0.05, self._control_loop)  # 20 Hz

        self.get_logger().info(
            f'Pure Pursuit ready | '
            f'L={self._L}m  lane={self._lw}m  '
            f'ld=[{self._ld_min:.1f}, {self._ld_max:.1f}]m  '
            f'ld_k={self._ld_k}  throttle={self._throttle}'
        )

    # ── Callbacks ──────────────────────────────────────────────────────────

    def _cb_center(self, msg: Float32):
        self._center_norm = float(msg.data) if msg.data >= 0.0 else None

    def _cb_odom(self, msg: Odometry):
        vx = msg.twist.twist.linear.x
        vy = msg.twist.twist.linear.y
        self._speed = math.sqrt(vx * vx + vy * vy)

    # ── Core Pure Pursuit ──────────────────────────────────────────────────

    def _lookahead(self) -> float:
        """Velocity-adaptive lookahead distance (Autoware formula, simplified)."""
        ld = self._ld_k * self._speed
        return float(max(self._ld_min, min(self._ld_max, ld)))

    def _pure_pursuit_steer(self, center_norm: float) -> float:
        """
        Returns normalised CARLA steer in [-1, 1].

        Derivation (from Autoware planning_utils.cpp):
          target lateral offset  y  = (center_norm - 0.5) * lane_width
          lookahead distance     ld = velocity-adaptive
          radius of curvature    R  = ld² / (2 * y)          [same as Autoware calcRadius]
          curvature              κ  = 1 / R = 2y / ld²
          steering angle         δ  = atan(L * κ)             [Autoware convertCurvatureToSteeringAngle]
          CARLA normalisation        δ / max_steer_rad → [-1, 1]
        """
        ld = self._lookahead()
        lateral_error = (center_norm - 0.5) * self._lw   # metres, + = right of centre

        kappa     = 2.0 * lateral_error / (ld ** 2)      # curvature [1/m]
        steer_rad = math.atan(self._L * kappa)            # steering angle [rad]
        steer_norm = steer_rad / self._max_steer          # normalise to [-1, 1]

        return max(-1.0, min(1.0, steer_norm)), ld, lateral_error, steer_rad

    # ── Control loop ───────────────────────────────────────────────────────

    def _control_loop(self):
        cmd = CarlaEgoVehicleControl()
        cmd.throttle      = self._throttle
        cmd.brake         = 0.0
        cmd.hand_brake    = False
        cmd.reverse       = False
        cmd.manual_gear_shift = False

        if self._center_norm is None:
            cmd.throttle = 0.0
            cmd.brake    = 1.0
            cmd.steer    = 0.0
            self.pub.publish(cmd)
            return

        cmd.steer, ld, lat_err, steer_rad = self._pure_pursuit_steer(self._center_norm)
        self.pub.publish(cmd)

        self.get_logger().info(
            f'speed={self._speed:.1f}m/s  ld={ld:.2f}m  '
            f'lat_err={lat_err:+.3f}m  '
            f'steer={math.degrees(steer_rad):+.1f}°  norm={cmd.steer:+.3f}'
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
