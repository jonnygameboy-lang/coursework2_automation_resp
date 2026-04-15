#!/usr/bin/env python3
"""
task4_compare_node.py

Task 4 comparison and visualisation node.

This node:
- subscribes to /odom for true state
- subscribes to /task4/noisy_pose
- subscribes to /task4/estimated_pose
- publishes RViz markers
- logs CSV for evidence generation

Published topic:
- /task4/markers
"""

import csv
import math
import os
import time

import rospy
from geometry_msgs.msg import Pose2D
from nav_msgs.msg import Odometry
from visualization_msgs.msg import Marker


def wrap_angle(angle_rad: float) -> float:
    """Wrap angle to [-pi, pi]."""
    return (angle_rad + math.pi) % (2.0 * math.pi) - math.pi


def quaternion_to_yaw(quaternion) -> float:
    """Convert quaternion to yaw."""
    x_q = quaternion.x
    y_q = quaternion.y
    z_q = quaternion.z
    w_q = quaternion.w

    siny_cosp = 2.0 * (w_q * z_q + x_q * y_q)
    cosy_cosp = 1.0 - 2.0 * (y_q * y_q + z_q * z_q)
    return math.atan2(siny_cosp, cosy_cosp)


class Task4CompareNode:
    """Visualise and compare true, noisy, and estimated state."""

    def __init__(self) -> None:
        rospy.init_node("task4_compare_node", anonymous=False)

        self.true_pose = None
        self.noisy_pose = None
        self.estimated_pose = None

        self.goal_x_m = rospy.get_param("~goal_x_m", 2.0)
        self.goal_y_m = rospy.get_param("~goal_y_m", 2.0)
        self.goal_theta_rad = rospy.get_param("~goal_theta_rad", 0.0)

        self.marker_pub = rospy.Publisher("/task4/markers", Marker, queue_size=20)

        rospy.Subscriber("/odom", Odometry, self.odom_callback)
        rospy.Subscriber("/task4/noisy_pose", Pose2D, self.noisy_callback)
        rospy.Subscriber("/task4/estimated_pose", Pose2D, self.estimated_callback)

        log_directory = os.path.expanduser("~/robot_ws/task4_logs")
        os.makedirs(log_directory, exist_ok=True)
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        self.csv_path = os.path.join(log_directory, f"task4_compare_log_{timestamp}.csv")

        self.csv_file = open(self.csv_path, "w", newline="", encoding="utf-8")
        self.csv_writer = csv.writer(self.csv_file)
        self.csv_writer.writerow([
            "time_s",
            "true_x_m", "true_y_m", "true_theta_rad",
            "noisy_x_m", "noisy_y_m", "noisy_theta_rad",
            "est_x_m", "est_y_m", "est_theta_rad",
            "goal_x_m", "goal_y_m", "goal_theta_rad",
            "position_error_m",
            "heading_error_rad",
        ])

        self.timer = rospy.Timer(rospy.Duration(0.1), self.timer_callback)

        rospy.loginfo("task4_compare_node started.")
        rospy.loginfo("CSV log file: %s", self.csv_path)

    def odom_callback(self, odom_msg: Odometry) -> None:
        """Store true pose from odometry."""
        pose = Pose2D()
        pose.x = odom_msg.pose.pose.position.x
        pose.y = odom_msg.pose.pose.position.y
        pose.theta = quaternion_to_yaw(odom_msg.pose.pose.orientation)
        self.true_pose = pose

    def noisy_callback(self, pose_msg: Pose2D) -> None:
        """Store noisy pose."""
        self.noisy_pose = pose_msg

    def estimated_callback(self, pose_msg: Pose2D) -> None:
        """Store estimated pose."""
        self.estimated_pose = pose_msg

    def make_marker(
        self,
        marker_id: int,
        ns: str,
        x_m: float,
        y_m: float,
        z_m: float,
        scale_m: float,
        r: float,
        g: float,
        b: float
    ) -> Marker:
        """Create a sphere marker for RViz."""
        marker = Marker()
        marker.header.frame_id = "odom"
        marker.header.stamp = rospy.Time.now()
        marker.ns = ns
        marker.id = marker_id
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD
        marker.pose.position.x = x_m
        marker.pose.position.y = y_m
        marker.pose.position.z = z_m
        marker.pose.orientation.w = 1.0
        marker.scale.x = scale_m
        marker.scale.y = scale_m
        marker.scale.z = scale_m
        marker.color.a = 1.0
        marker.color.r = r
        marker.color.g = g
        marker.color.b = b
        return marker

    def timer_callback(self, _event) -> None:
        """Publish markers and log data."""
        if self.true_pose is None or self.noisy_pose is None or self.estimated_pose is None:
            return

        true_marker = self.make_marker(
            0, "true_pose", self.true_pose.x, self.true_pose.y, 0.10, 0.18, 0.0, 0.0, 1.0
        )
        noisy_marker = self.make_marker(
            1, "noisy_pose", self.noisy_pose.x, self.noisy_pose.y, 0.10, 0.12, 1.0, 0.0, 0.0
        )
        estimated_marker = self.make_marker(
            2, "estimated_pose", self.estimated_pose.x, self.estimated_pose.y, 0.10, 0.15, 0.0, 1.0, 0.0
        )
        goal_marker = self.make_marker(
            3, "goal_pose", self.goal_x_m, self.goal_y_m, 0.10, 0.22, 1.0, 0.5, 0.0
        )

        self.marker_pub.publish(true_marker)
        self.marker_pub.publish(noisy_marker)
        self.marker_pub.publish(estimated_marker)
        self.marker_pub.publish(goal_marker)

        position_error_m = math.sqrt(
            (self.estimated_pose.x - self.true_pose.x) ** 2 +
            (self.estimated_pose.y - self.true_pose.y) ** 2
        )
        heading_error_rad = wrap_angle(self.estimated_pose.theta - self.true_pose.theta)

        self.csv_writer.writerow([
            rospy.get_time(),
            self.true_pose.x, self.true_pose.y, self.true_pose.theta,
            self.noisy_pose.x, self.noisy_pose.y, self.noisy_pose.theta,
            self.estimated_pose.x, self.estimated_pose.y, self.estimated_pose.theta,
            self.goal_x_m, self.goal_y_m, self.goal_theta_rad,
            position_error_m,
            heading_error_rad,
        ])
        self.csv_file.flush()


if __name__ == "__main__":
    try:
        Task4CompareNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass