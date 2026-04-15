"""
plotting.py

Plot generation for ENGM020 Coursework 2, Part 2, Task 1.

This module creates publication-style figures for the coursework slides and
saves them to the outputs/figures directory.
"""

from __future__ import annotations

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt


def create_task1_figures(results: dict, output_directory: str | Path) -> None:
    """
    Create and save all Task 1 figures.

    Parameters
    ----------
    results : dict
        Simulation results dictionary.
    output_directory : str | Path
        Directory in which to save the figure files.
    """
    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)

    time_s = results["time_s"]
    true_states = results["true_states"]
    gps_measurements = results["gps_measurements"]
    steering_angles_rad = results["steering_angles_rad"]

    x_true_m = true_states[:, 0]
    y_true_m = true_states[:, 1]
    heading_true_rad = true_states[:, 2]

    x_gps_m = gps_measurements[:, 0]
    y_gps_m = gps_measurements[:, 1]

    x_error_m = x_gps_m - x_true_m
    y_error_m = y_gps_m - y_true_m
    steering_angle_deg = np.rad2deg(steering_angles_rad)
    heading_true_deg = np.rad2deg(heading_true_rad)

    # Figure 1: 2D trajectory
    plt.figure(figsize=(8, 6))
    plt.plot(x_true_m, y_true_m, label="True vehicle trajectory", linewidth=2)
    plt.scatter(x_gps_m, y_gps_m, label="Noisy GPS measurements", s=12, alpha=0.7)
    plt.xlabel("x position (m)")
    plt.ylabel("y position (m)")
    plt.title("Task 1: Vehicle trajectory and noisy GPS measurements")
    plt.axis("equal")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_directory / "task1_figure1_trajectory.png", dpi=300)
    plt.close()

    # Figure 2: State histories
    plt.figure(figsize=(9, 6))
    plt.plot(time_s, x_true_m, label="x position")
    plt.plot(time_s, y_true_m, label="y position")
    plt.plot(time_s, heading_true_deg, label="heading angle")
    plt.xlabel("Time (s)")
    plt.ylabel("State value")
    plt.title("Task 1: Vehicle state histories")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_directory / "task1_figure2_state_histories.png", dpi=300)
    plt.close()

    # Figure 3: Measurement error
    plt.figure(figsize=(9, 5))
    plt.plot(time_s, x_error_m, label="x measurement error")
    plt.plot(time_s, y_error_m, label="y measurement error")
    plt.xlabel("Time (s)")
    plt.ylabel("Error (m)")
    plt.title("Task 1: GPS measurement error relative to true position")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_directory / "task1_figure3_measurement_error.png", dpi=300)
    plt.close()

    # Figure 4: Steering input history
    plt.figure(figsize=(9, 5))
    plt.plot(time_s, steering_angle_deg, linewidth=2)
    plt.xlabel("Time (s)")
    plt.ylabel("Steering angle (deg)")
    plt.title("Task 1: Applied steering input profile")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(output_directory / "task1_figure4_steering_profile.png", dpi=300)
    plt.close()