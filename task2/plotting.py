"""
plotting.py

Plot generation for ENGM020 Coursework 2, Part 2, Task 2.
"""

from __future__ import annotations

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt


def create_single_run_figures(results: dict, output_directory: str | Path) -> None:
    """
    Create figures for a single baseline Task 2 run.

    Parameters
    ----------
    results : dict
        Single-run Task 2 results.
    output_directory : str | Path
        Output directory for figure files.
    """
    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)

    reference_path = results["reference_path"]
    state_history = results["state_history"]
    lookahead_point_history = results["lookahead_point_history"]
    time_s = results["time_s"]
    steering_angle_deg = np.rad2deg(results["steering_history_rad"])
    cross_track_error_m = results["cross_track_error_history_m"]
    goal_point_xy_m = results["goal_point_xy_m"]

    # Figure 1: reference path and tracked path
    plt.figure(figsize=(8, 6))
    plt.plot(reference_path[:, 0], reference_path[:, 1], "--", linewidth=2, label="Reference path")
    plt.plot(state_history[:, 0], state_history[:, 1], linewidth=2, label="Tracked vehicle path")
    plt.scatter(goal_point_xy_m[0], goal_point_xy_m[1], s=60, label="Goal point")
    plt.xlabel("x position (m)")
    plt.ylabel("y position (m)")
    plt.title("Task 2: Reference path and tracked vehicle path")
    plt.axis("equal")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_directory / "task2_figure1_path_tracking.png", dpi=300)
    plt.close()

    # Figure 2: cross-track error
    plt.figure(figsize=(9, 5))
    plt.plot(time_s, cross_track_error_m, linewidth=2)
    plt.xlabel("Time (s)")
    plt.ylabel("Cross-track error (m)")
    plt.title("Task 2: Cross-track error history")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(output_directory / "task2_figure2_cross_track_error.png", dpi=300)
    plt.close()

    # Figure 3: steering angle
    plt.figure(figsize=(9, 5))
    plt.plot(time_s, steering_angle_deg, linewidth=2)
    plt.xlabel("Time (s)")
    plt.ylabel("Steering angle (deg)")
    plt.title("Task 2: Steering angle history")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(output_directory / "task2_figure3_steering_history.png", dpi=300)
    plt.close()

    # Figure 4: representative lookahead points along the run
    sample_indices = np.linspace(0, len(time_s) - 1, min(12, len(time_s)), dtype=int)

    plt.figure(figsize=(8, 6))
    plt.plot(reference_path[:, 0], reference_path[:, 1], "--", linewidth=2, label="Reference path")
    plt.plot(state_history[:, 0], state_history[:, 1], linewidth=2, label="Tracked path")
    plt.scatter(
        lookahead_point_history[sample_indices, 0],
        lookahead_point_history[sample_indices, 1],
        s=40,
        label="Sampled lookahead points"
    )
    plt.xlabel("x position (m)")
    plt.ylabel("y position (m)")
    plt.title("Task 2: Lookahead points used by the controller")
    plt.axis("equal")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_directory / "task2_figure4_lookahead_points.png", dpi=300)
    plt.close()


def create_lookahead_comparison_figures(results_list: list[dict], output_directory: str | Path) -> None:
    """
    Create comparison figures for different lookahead distances.

    Parameters
    ----------
    results_list : list[dict]
        Results for several lookahead distances.
    output_directory : str | Path
        Output directory.
    """
    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)

    # Figure 5: path comparison across lookahead distances
    plt.figure(figsize=(8, 6))
    reference_path = results_list[0]["reference_path"]
    plt.plot(reference_path[:, 0], reference_path[:, 1], "k--", linewidth=2, label="Reference path")

    for results in results_list:
        lookahead_distance_m = results["lookahead_distance_m"]
        state_history = results["state_history"]
        plt.plot(
            state_history[:, 0],
            state_history[:, 1],
            linewidth=2,
            label=f"Tracked path, Ld = {lookahead_distance_m:.1f} m"
        )

    plt.xlabel("x position (m)")
    plt.ylabel("y position (m)")
    plt.title("Task 2: Effect of lookahead distance on path tracking")
    plt.axis("equal")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_directory / "task2_figure5_lookahead_path_comparison.png", dpi=300)
    plt.close()

    # Figure 6: cross-track error comparison
    plt.figure(figsize=(9, 5))
    for results in results_list:
        lookahead_distance_m = results["lookahead_distance_m"]
        plt.plot(
            results["time_s"],
            results["cross_track_error_history_m"],
            linewidth=2,
            label=f"Ld = {lookahead_distance_m:.1f} m"
        )

    plt.xlabel("Time (s)")
    plt.ylabel("Cross-track error (m)")
    plt.title("Task 2: Cross-track error for different lookahead distances")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_directory / "task2_figure6_lookahead_error_comparison.png", dpi=300)
    plt.close()