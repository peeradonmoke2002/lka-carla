#!/usr/bin/python3

import math
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, DurabilityPolicy
from nav_msgs.msg import Odometry
from std_msgs.msg import Bool, String
from carla_msgs.msg import CarlaEgoVehicleControl
from lka_msgs.msg import LaneCenter

# State values published on /lka/controller/state
STATE_IDLE         = 'idle'          # waiting for first lane detection
STATE_DRIVING      = 'driving'       # lane detected, vehicle moving toward goal
STATE_GOAL_REACHED = 'goal_reached'  # vehicle passed stop_x, braking


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
        self.declare_parameter('stop_x',          108.0)   # stop when odometry x < this
        self.declare_parameter('center_bias_offset', 0.0)

        self.wheel_base  = self.get_parameter('wheel_base').value
        self.lane_width  = self.get_parameter('lane_width').value
        self.ld_min      = self.get_parameter('min_lookahead').value
        self.ld_max      = self.get_parameter('max_lookahead').value
        self.ld_k        = self.get_parameter('ld_velocity_ratio').value
        self.max_steer   = self.get_parameter('max_steer_rad').value
        self.throttle    = self.get_parameter('throttle').value
        self.stop_x      = self.get_parameter('stop_x').value
        self.center_bias = self.get_parameter('center_bias_offset').value

        self.center_norm: float | None = None
        self.speed:  float       = 0.0
        self.pos_x:  float | None = None
        self.state:  str          = STATE_IDLE

        self.create_subscription(LaneCenter, '/lka/lane_center', self.lane_center_callback, 10)
        self.create_subscription(Odometry, '/carla/ego_vehicle/odometry', self.odom_callback, 10)

        self.pub = self.create_publisher(
            CarlaEgoVehicleControl, '/carla/ego_vehicle/vehicle_control_cmd', 10)
        self.state_pub = self.create_publisher(
            String, '/lka/controller/state', 10)

        # Disable CARLA manual-control override so the bridge accepts our commands.
        # TRANSIENT_LOCAL matches the bridge subscriber's QoS.
        override_pub = self.create_publisher(
            Bool, '/carla/ego_vehicle/vehicle_control_manual_override',
            QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL))
        override_pub.publish(Bool(data=False))

        self.create_timer(0.05, self.control_loop)  # 20 Hz

        self.get_logger().info(
            f'Pure Pursuit ready | '
            f'L={self.wheel_base}m  lane={self.lane_width}m  '
            f'ld=[{self.ld_min:.1f}, {self.ld_max:.1f}]m  '
            f'ld_k={self.ld_k}  throttle={self.throttle}  '
            f'stop_x={self.stop_x}  center_bias={self.center_bias:+.4f}'
        )

    # ── Callbacks ──────────────────────────────────────────────────────────────

    def lane_center_callback(self, msg: LaneCenter):
        if msg.detected:
            self.center_norm = float(msg.center) - self.center_bias
        else:
            self.center_norm = None

    def odom_callback(self, msg: Odometry):
        vx = msg.twist.twist.linear.x
        vy = msg.twist.twist.linear.y
        self.speed = math.sqrt(vx * vx + vy * vy)
        self.pos_x = msg.pose.pose.position.x

    # ── Core Pure Pursuit ──────────────────────────────────────────────────────

    def lookahead(self) -> float:
        ld = self.ld_k * self.speed
        return float(max(self.ld_min, min(self.ld_max, ld)))

    def compute_steer(self, center_norm: float) -> tuple:
        ld = self.lookahead()
        lateral_error = (center_norm - 0.5) * self.lane_width
        kappa     = 2.0 * lateral_error / (ld ** 2)
        steer_rad = math.atan(self.wheel_base * kappa)
        steer_norm = steer_rad / self.max_steer
        return max(-1.0, min(1.0, steer_norm)), ld, lateral_error, steer_rad

    # ── Control loop ───────────────────────────────────────────────────────────

    def control_loop(self):
        cmd = CarlaEgoVehicleControl()
        cmd.hand_brake        = False
        cmd.reverse           = False
        cmd.manual_gear_shift = False

        # ── Goal reached: stop and hold ────────────────────────────────────────
        if self.pos_x is not None and self.pos_x < self.stop_x:
            self.state   = STATE_GOAL_REACHED
            cmd.throttle = 0.0
            cmd.brake    = 1.0
            cmd.steer    = 0.0
            self.pub.publish(cmd)
            self.state_pub.publish(String(data=self.state))
            self.get_logger().info(
                f'[{self.state}] x={self.pos_x:.2f} < stop_x={self.stop_x}',
                throttle_duration_sec=2.0,
            )
            return

        # ── No lane detection: idle / brake ────────────────────────────────────
        if self.center_norm is None:
            self.state   = STATE_IDLE
            cmd.throttle = 0.0
            cmd.brake    = 1.0
            cmd.steer    = 0.0
            self.pub.publish(cmd)
            self.state_pub.publish(String(data=self.state))
            return

        # ── Lane detected: drive ───────────────────────────────────────────────
        self.state = STATE_DRIVING
        cmd.throttle = self.throttle
        cmd.brake    = 0.0
        cmd.steer, ld, lat_err, steer_rad = self.compute_steer(self.center_norm)
        self.pub.publish(cmd)
        self.state_pub.publish(String(data=self.state))

        x_str = f'{self.pos_x:.1f}' if self.pos_x is not None else 'n/a'
        self.get_logger().info(
            f'[{self.state}]  x={x_str}  speed={self.speed:.1f}m/s  '
            f'ld={ld:.2f}m  lat_err={lat_err:+.3f}m  '
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
