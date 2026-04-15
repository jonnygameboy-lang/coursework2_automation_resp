"""
plotting.py

Plot generation for ENGM020 Coursework 2, Part 2, Task 3.
"""

from __future__ import annotations

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt


def create_task3_figures(results: dict, output_directory: str | Path) -> None:
    """
    Create and save all Task 3 figures.

    Parameters
    ----------
    results : dict
        Task 3 results dictionary.
    output_directory : str | Path
        Output directory.
    """
    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)

    time_s = results["time_s"]
    true_states = results["true_state_history"]
    measurements = results["measurement_history"]
    kf_states = results["kf_state_history"]
    ekf_states = results["ekf_state_history"]

    measurement_errors = results["measurement_errors"]
    kf_errors = results["kf_errors"]
    ekf_errors = results["ekf_errors"]

    reference_path = results["reference_path"]
    goal_point_xy_m = results["goal_point_xy_m"]

    # Figure 1: True, measured, KF, EKF trajectory
    plt.figure(figsize=(8, 6))
    plt.plot(reference_path[:, 0], reference_path[:, 1], "k--", linewidth=2, label="Reference path")
    plt.plot(true_states[:, 0], true_states[:, 1], linewidth=2, label="True trajectory")
    plt.scatter(measurements[:, 0], measurements[:, 1], s=12, alpha=0.5, label="Noisy measurements")
    plt.plot(kf_states[:, 0], kf_states[:, 1], linewidth=2, label="KF estimate")
    plt.plot(ekf_states[:, 0], ekf_states[:, 1], linewidth=2, label="EKF estimate")
    plt.scatter(goal_point_xy_m[0], goal_point_xy_m[1], s=60, label="Goal point")
    plt.xlabel("x position (m)")
    plt.ylabel("y position (m)")
    plt.title("Task 3: True, measured, KF, and EKF trajectories")
    plt.axis("equal")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_directory / "task3_figure1_trajectory_comparison.png", dpi=300)
    plt.close()

    # Figure 2: Position error norm
    measurement_position_error = np.linalg.norm(measurement_errors[:, :2], axis=1)
    kf_position_error = np.linalg.norm(kf_errors[:, :2], axis=1)
    ekf_position_error = np.linalg.norm(ekf_errors[:, :2], axis=1)

    plt.figure(figsize=(9, 5))
    plt.plot(time_s, measurement_position_error, linewidth=2, label="Measurement error")
    plt.plot(time_s, kf_position_error, linewidth=2, label="KF position error")
    plt.plot(time_s, ekf_position_error, linewidth=2, label="EKF position error")
    plt.xlabel("Time (s)")
    plt.ylabel("Position error norm (m)")
    plt.title("Task 3: Position estimation error comparison")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_directory / "task3_figure2_position_error_comparison.png", dpi=300)
    plt.close()

    # Figure 3: Heading error comparison
    measurement_heading_error_deg = np.rad2deg(measurement_errors[:, 2])
    kf_heading_error_deg = np.rad2deg(kf_errors[:, 2])
    ekf_heading_error_deg = np.rad2deg(ekf_errors[:, 2])

    plt.figure(figsize=(9, 5))
    plt.plot(time_s, measurement_heading_error_deg, linewidth=2, label="Measurement heading error")
    plt.plot(time_s, kf_heading_error_deg, linewidth=2, label="KF heading error")
    plt.plot(time_s, ekf_heading_error_deg, linewidth=2, label="EKF heading error")
    plt.xlabel("Time (s)")
    plt.ylabel("Heading error (deg)")
    plt.title("Task 3: Heading estimation error comparison")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_directory / "task3_figure3_heading_error_comparison.png", dpi=300)
    plt.close()

    # Figure 4: x and y estimation comparison
    plt.figure(figsize=(9, 5))
    plt.plot(time_s, true_states[:, 0], linewidth=2, label="True x")
    plt.plot(time_s, measurements[:, 0], linestyle=":", linewidth=1.5, label="Measured x")
    plt.plot(time_s, kf_states[:, 0], linewidth=2, label="KF x")
    plt.plot(time_s, ekf_states[:, 0], linewidth=2, label="EKF x")
    plt.xlabel("Time (s)")
    plt.ylabel("x position (m)")
    plt.title("Task 3: x-position estimate comparison")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_directory / "task3_figure4_x_state_comparison.png", dpi=300)
    plt.close()

    plt.figure(figsize=(9, 5))
    plt.plot(time_s, true_states[:, 1], linewidth=2, label="True y")
    plt.plot(time_s, measurements[:, 1], linestyle=":", linewidth=1.5, label="Measured y")
    plt.plot(time_s, kf_states[:, 1], linewidth=2, label="KF y")
    plt.plot(time_s, ekf_states[:, 1], linewidth=2, label="EKF y")
    plt.xlabel("Time (s)")
    plt.ylabel("y position (m)")
    plt.title("Task 3: y-position estimate comparison")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_directory / "task3_figure5_y_state_comparison.png", dpi=300)
    plt.close()