#!/usr/bin/python3
"""
Save one RGB + semantic segmentation image pair to /tmp for inspection.
Run while carla_ros_bridge is active.

Usage:
  ros2 run lka_dataset_collection save_sem_sample.py
"""

import rclpy
import rclpy.parameter
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import message_filters
import cv2
import numpy as np


class SaveSemSample(Node):

    def __init__(self):
        super().__init__('save_sem_sample')
        self.set_parameters([rclpy.parameter.Parameter(
            'use_sim_time', rclpy.Parameter.Type.BOOL, True)])

        self.bridge = CvBridge()
        self.done   = False

        rgb_sub = message_filters.Subscriber(
            self, Image, '/carla/ego_vehicle/CAM_FRONT/image')
        sem_sub = message_filters.Subscriber(
            self, Image, '/carla/ego_vehicle/semantic_segmentation_front/image')

        self.sync = message_filters.ApproximateTimeSynchronizer(
            [rgb_sub, sem_sub], queue_size=10, slop=0.05)
        self.sync.registerCallback(self._callback)

        self.get_logger().info('Waiting for camera frames...')

    def _callback(self, rgb_msg, sem_msg):
        if self.done:
            return

        rgb = self.bridge.imgmsg_to_cv2(rgb_msg, desired_encoding='bgr8')
        sem = self.bridge.imgmsg_to_cv2(sem_msg, desired_encoding='bgr8')
        h, w = rgb.shape[:2]

        # Side-by-side: RGB left, semantic right (resize sem to same height)
        combined = np.hstack([rgb, sem])

        out_dir = '/home/peeradon/lka_dataset'
        cv2.imwrite(f'{out_dir}/sample_rgb.jpg', rgb)
        cv2.imwrite(f'{out_dir}/sample_sem.jpg', sem)
        cv2.imwrite(f'{out_dir}/sample_combined.jpg', combined)

        self.get_logger().info(
            f'Saved!  size={w}x{h}\n'
            f'  {out_dir}/sample_rgb.jpg      — RGB camera\n'
            f'  {out_dir}/sample_sem.jpg      — semantic segmentation\n'
            f'  {out_dir}/sample_combined.jpg — side by side')

        self.done = True


def main(args=None):
    rclpy.init(args=args)
    node = SaveSemSample()
    while rclpy.ok() and not node.done:
        rclpy.spin_once(node, timeout_sec=0.1)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
