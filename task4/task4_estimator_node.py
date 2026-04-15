#!/usr/bin/env python3
"""
task4_estimator_node.py

Task 4 estimator node.

This node:
- subscribes to /odom for the true robot pose from Gazebo
- subscribes to /cmd_vel for the applied control input
- internally generates a noisy measurement
- runs an EKF
- publishes the noisy measurement and estimated pose

Published topics:
- /task4/noisy_pose
- /task4/estimated_pose
"""

import math
import random
import numpy as np
import rospy
from geometry_msgs.msg import Pose2D, Twist
from nav_msgs.msg import Odometry


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


class Task4EstimatorNode:
    """Single EKF estimator node for Task 4."""

    def __init__(self) -> None:
        rospy.init_node("task4_estimator_node", anonymous=False)

        self.time_step_s = rospy.get_param("~time_step_s", 0.1)

        self.position_noise_std_m = rospy.get_param("~position_noise_std_m", 0.5)
        self.heading_noise_std_deg = rospy.get_param("~heading_noise_std_deg", 5.0)
        self.heading_noise_std_rad = math.radians(self.heading_noise_std_deg)

        process_noise_position = rospy.get_param("~process_noise_position", 0.10)
        process_noise_heading_deg = rospy.get_param("~process_noise_heading_deg", 2.0)
        measurement_noise_position = rospy.get_param("~measurement_noise_position", 0.50)
        measurement_noise_heading_deg = rospy.get_param("~measurement_noise_heading_deg", 5.0)
        initial_position_std = rospy.get_param("~initial_position_std", 1.0)
        initial_heading_std_deg = rospy.get_param("~initial_heading_std_deg", 20.0)

        process_noise_heading_rad = math.radians(process_noise_heading_deg)
        measurement_noise_heading_rad = math.radians(measurement_noise_heading_deg)
        initial_heading_std_rad = math.radians(initial_heading_std_deg)

        self.Q = np.diag([
            process_noise_position**2,
            process_noise_position**2,
            process_noise_heading_rad**2,
        ])

        self.R = np.diag([
            measurement_noise_position**2,
            measurement_noise_position**2,
            measurement_noise_heading_rad**2,
        ])

        self.P = np.diag([
            initial_position_std**2,
            initial_position_std**2,
            initial_heading_std_rad**2,
        ])

        self.true_state = None
        self.noisy_measurement = None
        self.x_est = np.zeros(3, dtype=float)
        self.control_input = np.zeros(2, dtype=float)
        self.is_initialised = False

        random.seed(42)

        self.noisy_pose_pub = rospy.Publisher("/task4/noisy_pose", Pose2D, queue_size=10)
        self.estimated_pose_pub = rospy.Publisher("/task4/estimated_pose", Pose2D, queue_size=10)

        rospy.Subscriber("/odom", Odometry, self.odom_callback)
        rospy.Subscriber("/cmd_vel", Twist, self.cmd_callback)

        self.timer = rospy.Timer(rospy.Duration(self.time_step_s), self.timer_callback)

        rospy.loginfo("task4_estimator_node started.")

    def cmd_callback(self, twist_msg: Twist) -> None:
        """Store control input [v, omega]."""
        self.control_input[0] = twist_msg.linear.x
        self.control_input[1] = twist_msg.angular.z

    def odom_callback(self, odom_msg: Odometry) -> None:
        """Read true state and generate noisy measurement."""
        true_x_m = odom_msg.pose.pose.position.x
        true_y_m = odom_msg.pose.pose.position.y
        true_theta_rad = quaternion_to_yaw(odom_msg.pose.pose.orientation)

        self.true_state = np.array([true_x_m, true_y_m, true_theta_rad], dtype=float)

        meas_x_m = true_x_m + random.gauss(0.0, self.position_noise_std_m)
        meas_y_m = true_y_m + random.gauss(0.0, self.position_noise_std_m)
        meas_theta_rad = wrap_angle(
            true_theta_rad + random.gauss(0.0, self.heading_noise_std_rad)
        )

        self.noisy_measurement = np.array([meas_x_m, meas_y_m, meas_theta_rad], dtype=float)

        meas_msg = Pose2D()
        meas_msg.x = meas_x_m
        meas_msg.y = meas_y_m
        meas_msg.theta = meas_theta_rad
        self.noisy_pose_pub.publish(meas_msg)

    def nonlinear_state_transition(self, state: np.ndarray, control_input: np.ndarray) -> np.ndarray:
        """Nonlinear planar motion model using [v, omega]."""
        x_m, y_m, theta_rad = state
        velocity_mps, yaw_rate_radps = control_input

        x_next_m = x_m + velocity_mps * math.cos(theta_rad) * self.time_step_s
        y_next_m = y_m + velocity_mps * math.sin(theta_rad) * self.time_step_s
        theta_next_rad = wrap_angle(theta_rad + yaw_rate_radps * self.time_step_s)

        return np.array([x_next_m, y_next_m, theta_next_rad], dtype=float)

    def compute_motion_jacobian(self, state: np.ndarray, control_input: np.ndarray) -> np.ndarray:
        """Compute motion Jacobian for the EKF."""
        theta_rad = state[2]
        velocity_mps = control_input[0]

        return np.array([
            [1.0, 0.0, -velocity_mps * math.sin(theta_rad) * self.time_step_s],
            [0.0, 1.0,  velocity_mps * math.cos(theta_rad) * self.time_step_s],
            [0.0, 0.0,  1.0],
        ])

    def ekf_step(self, measurement: np.ndarray) -> None:
        """Run one EKF predict-correct step."""
        H = np.eye(3)

        x_pred = self.nonlinear_state_transition(self.x_est, self.control_input)
        F = self.compute_motion_jacobian(self.x_est, self.control_input)
        P_pred = F @ self.P @ F.T + self.Q

        innovation = measurement - H @ x_pred
        innovation[2] = wrap_angle(innovation[2])

        S = H @ P_pred @ H.T + self.R
        K = P_pred @ H.T @ np.linalg.inv(S)

        self.x_est = x_pred + K @ innovation
        self.x_est[2] = wrap_angle(self.x_est[2])

        self.P = (np.eye(3) - K @ H) @ P_pred

    def timer_callback(self, _event) -> None:
        """Periodic EKF update and estimate publication."""
        if self.noisy_measurement is None:
            return

        if not self.is_initialised:
            self.x_est = self.noisy_measurement.copy()
            self.is_initialised = True
        else:
            self.ekf_step(self.noisy_measurement)

        est_msg = Pose2D()
        est_msg.x = float(self.x_est[0])
        est_msg.y = float(self.x_est[1])
        est_msg.theta = float(self.x_est[2])

        self.estimated_pose_pub.publish(est_msg)


if __name__ == "__main__":
    try:
        Task4EstimatorNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass