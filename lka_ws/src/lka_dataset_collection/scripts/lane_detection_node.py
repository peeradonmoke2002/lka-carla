#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Float32
from cv_bridge import CvBridge
from ultralytics import YOLO
import cv2
import numpy as np


WEIGHTS = "/home/peeradon/lka-carla-yolo/lka_ws/src/lka_dataset_collection/scripts/runs/segment/training/runs/lka_seg_l5/weights/best.pt"


class LaneDetectionNode(Node):
    def __init__(self):
        super().__init__("lane_detection_node")

        self.declare_parameter("weights", WEIGHTS)
        self.declare_parameter("conf_threshold", 0.25)
        self.declare_parameter("weather_mode", "clear")  # clear | fog | rain | night

        weights = self.get_parameter("weights").get_parameter_value().string_value
        self.conf = self.get_parameter("conf_threshold").get_parameter_value().double_value
        self.weather_mode = self.get_parameter("weather_mode").get_parameter_value().string_value

        self.model = YOLO(weights)
        self.bridge = CvBridge()

        self.sub = self.create_subscription(
            Image,
            "/carla/ego_vehicle/CAM_FRONT/image",
            self.image_callback,
            10,
        )
        self.pub_enhanced = self.create_publisher(Image, "/lka/enhanced_image", 10)
        self.pub_degraded = self.create_publisher(Image, "/lka/degraded_image", 10)
        self.pub_center = self.create_publisher(Float32, "/lka/lane_center", 10)
        self.pub_conf = self.create_publisher(Float32, "/lka/detection_confidence", 10)
        # before-preprocessing inference (for paper comparison)
        self.pub_degraded_det = self.create_publisher(Image, "/lka/degraded_detection", 10)
        self.pub_conf_degraded = self.create_publisher(Float32, "/lka/detection_confidence_degraded", 10)

        self.get_logger().info(
            f"Lane detection node ready | weather: {self.weather_mode} | weights: {weights}"
        )

    # ------------------------------------------------------------------
    @staticmethod
    def _gamma_lut(gamma: float) -> np.ndarray:
        return np.array(
            [min(255, int((i / 255.0) ** (1.0 / gamma) * 255)) for i in range(256)],
            dtype=np.uint8,
        )

    def degrade(self, img: np.ndarray) -> np.ndarray:
        """Synthetically degrade image to simulate real-world weather conditions."""
        if self.weather_mode == "fog":
            # blend with white haze + gaussian blur
            white = np.full_like(img, 220)
            hazy = cv2.addWeighted(img, 0.45, white, 0.55, 0)
            return cv2.GaussianBlur(hazy, (21, 21), 0)

        elif self.weather_mode == "rain":
            # darken + gaussian noise to simulate wet lens / rain streaks
            dark = (img * 0.65).astype(np.uint8)
            noise = np.random.normal(0, 18, img.shape).astype(np.int16)
            return np.clip(dark.astype(np.int16) + noise, 0, 255).astype(np.uint8)

        elif self.weather_mode == "night":
            # strong darkness — simulate low ambient light
            return (img * 0.18).astype(np.uint8)

        return img  # clear — no degradation

    def preprocess(self, img: np.ndarray) -> np.ndarray:
        """Enhance degraded image to recover lane visibility."""
        if self.weather_mode == "fog":
            # CLAHE on L channel to recover contrast lost in haze
            lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            lab = cv2.merge([clahe.apply(l), a, b])
            return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

        elif self.weather_mode == "rain":
            # bilateral filter removes noise/streaks while keeping lane edges sharp
            denoised = cv2.bilateralFilter(img, d=9, sigmaColor=75, sigmaSpace=75)
            return self._gamma_lut(1.3)[denoised]

        elif self.weather_mode == "night":
            # strong gamma brighten to recover lanes in darkness
            return self._gamma_lut(1.8)[img]

        return img  # clear

    # ------------------------------------------------------------------
    def image_callback(self, msg: Image):
        img = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        h, w = img.shape[:2]

        # Step 1: synthetic degradation (simulates real-world weather)
        degraded = self.degrade(img)

        # Step 2: YOLO on degraded image (no preprocessing — baseline)
        results_deg = self.model.predict(degraded, conf=self.conf, device=0, verbose=False)[0]
        lane_center_deg, conf_deg = self._extract_lane_info(results_deg)

        annotated_deg = results_deg.plot()
        # self._draw_center_overlay(annotated_deg, lane_center_deg, conf_deg, w, h)
        self.pub_degraded.publish(self.bridge.cv2_to_imgmsg(degraded, encoding="bgr8"))
        self.pub_degraded_det.publish(self.bridge.cv2_to_imgmsg(annotated_deg, encoding="bgr8"))
        conf_deg_msg = Float32()
        conf_deg_msg.data = float(conf_deg)
        self.pub_conf_degraded.publish(conf_deg_msg)

        # Step 3: preprocessing to recover visibility
        enhanced = self.preprocess(degraded)

        # Step 4: YOLO on enhanced image
        results = self.model.predict(enhanced, conf=self.conf, device=0, verbose=False)[0]
        lane_center_x, mean_conf = self._extract_lane_info(results)

        annotated = results.plot()
        # self._draw_center_overlay(annotated, lane_center_x, mean_conf, w, h)
        self.pub_enhanced.publish(self.bridge.cv2_to_imgmsg(annotated, encoding="bgr8"))

        center_msg = Float32()
        center_msg.data = float(lane_center_x / w) if lane_center_x is not None else -1.0
        self.pub_center.publish(center_msg)

        conf_msg = Float32()
        conf_msg.data = float(mean_conf)
        self.pub_conf.publish(conf_msg)

        self.get_logger().info(
            f"[{self.weather_mode}] "
            f"conf_degraded={conf_deg:.3f}  conf_enhanced={mean_conf:.3f}  "
            f"delta={mean_conf - conf_deg:+.3f}  center={center_msg.data:.3f}"
        )

    # ------------------------------------------------------------------
    @staticmethod
    def _draw_center_overlay(img: np.ndarray, lane_center_x, conf: float, w: int, h: int):
        img_cx = w // 2
        bottom_y = h - 10
        top_y = h // 2

        # image center reference line always visible (white dashed)
        # for y in range(top_y, bottom_y, 20):
        #     cv2.line(img, (img_cx, y), (img_cx, min(y + 10, bottom_y)), (255, 255, 255), 1)

        if lane_center_x is None:
            # red semi-transparent banner
            overlay = img.copy()
            cv2.rectangle(overlay, (0, 0), (w, 95), (0, 0, 180), -1)
            cv2.addWeighted(overlay, 0.45, img, 0.55, 0, img)
            cv2.putText(img, "!! NO DETECTION !!", (10, 32),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
            cv2.putText(img, "center: ---   error: ---", (10, 58),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
            cv2.putText(img, f"conf:   {conf:.3f}", (10, 84),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
            return

        cx = int(lane_center_x)
        lateral_error = cx - img_cx

        # lane center vertical line (green = within ±50px, red = outside)
        color = (0, 255, 0) if abs(lateral_error) < 50 else (0, 0, 255)
        cv2.line(img, (cx, top_y), (cx, bottom_y), color, 2)

        # dot at center
        cv2.circle(img, (cx, (top_y + bottom_y) // 2), 6, color, -1)

        # horizontal arrow from image center to lane center
        cv2.arrowedLine(img, (img_cx, top_y + 20), (cx, top_y + 20), (0, 255, 255), 2, tipLength=0.2)

        # text overlay
        error_norm = lateral_error / w
        cv2.putText(img, f"center: {lane_center_x / w:.3f}", (10, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(img, f"error:  {error_norm:+.3f}", (10, 54),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        cv2.putText(img, f"conf:   {conf:.3f}", (10, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

    # ------------------------------------------------------------------
    def _extract_lane_info(self, results):
        """Return (lane_center_x_px, mean_confidence)."""
        if results.masks is None or len(results.boxes) == 0:
            return None, 0.0

        boxes = results.boxes
        masks = results.masks.xy  # list of polygon arrays

        left_xs, right_xs = [], []
        confs = boxes.conf.cpu().numpy()
        class_ids = boxes.cls.cpu().numpy().astype(int)
        # class 0 = left_marking, class 1 = right_edge (from data.yaml)

        for poly, cls in zip(masks, class_ids):
            if len(poly) == 0:
                continue
            cx = float(poly[:, 0].mean())
            if cls == 0:
                left_xs.append(cx)
            else:
                right_xs.append(cx)

        if left_xs and right_xs:
            lane_center = (np.mean(left_xs) + np.mean(right_xs)) / 2.0
        elif left_xs:
            lane_center = np.mean(left_xs)
        elif right_xs:
            lane_center = np.mean(right_xs)
        else:
            return None, 0.0

        return lane_center, float(confs.mean())


def main(args=None):
    rclpy.init(args=args)
    node = LaneDetectionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
