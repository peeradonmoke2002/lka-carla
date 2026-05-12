#!/usr/bin/python3

import math
import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from std_msgs.msg import Float64

XODR_PATH = '/home/peeradon/carla/CarlaUE4/Content/Carla/Maps/OpenDrive/Town01.xodr'


class GTNode(Node):
    def __init__(self):
        super().__init__('gt_node')

        self.declare_parameter('xodr_path', XODR_PATH)
        xodr_path = self.get_parameter('xodr_path').value

        self._load_map(xodr_path)

        self.create_subscription(Odometry, '/carla/ego_vehicle/odometry',
                                 self._odom_cb, 10)
        self.pub_cte = self.create_publisher(Float64, '/lka/gt/cross_track_m', 10)

        self.get_logger().info(f'GT node ready | map: {xodr_path}')

    # ── Map loading ─────────────────────────────────────────────────────

    def _load_map(self, xodr_path: str):
        import carla
        with open(xodr_path) as f:
            xodr = f.read()
        self._carla_map = carla.Map('Town01', xodr)
        self.get_logger().info('Town01 OpenDrive map loaded (offline)')

    # ── Odometry callback ───────────────────────────────────────────────

    def _odom_cb(self, msg: Odometry):
        import carla

        # ROS bridge flips Y: carla_y = -ros_y
        ros_x = msg.pose.pose.position.x
        ros_y = msg.pose.pose.position.y
        ros_z = msg.pose.pose.position.z

        ego_loc = carla.Location(x=ros_x, y=-ros_y, z=ros_z)

        wp = self._carla_map.get_waypoint(
            ego_loc,
            project_to_road=True,
            lane_type=carla.LaneType.Driving,
        )
        if wp is None:
            return

        lc = wp.transform.location   # lane centre in CARLA coords

        # Signed CTE: positive = ego is to the right of lane centre
        # Use the waypoint's right vector to determine sign
        wp_fwd = wp.transform.get_forward_vector()
        dx = ego_loc.x - lc.x
        dy = ego_loc.y - lc.y
        # right vector = rotate forward 90° clockwise in XY plane
        right_x =  wp_fwd.y
        right_y = -wp_fwd.x
        cte_signed = dx * right_x + dy * right_y

        out = Float64()
        out.data = float(cte_signed)
        self.pub_cte.publish(out)

        self.get_logger().info(
            f'CTE={cte_signed:+.3f}m  ego=({ros_x:.1f},{ros_y:.1f})  '
            f'lane_centre=({lc.x:.1f},{-lc.y:.1f})',
            throttle_duration_sec=1.0,
        )


def main(args=None):
    rclpy.init(args=args)
    node = GTNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
