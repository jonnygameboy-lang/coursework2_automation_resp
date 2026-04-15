"""
simulation.py

Closed-loop simulation for ENGM020 Coursework 2, Part 2, Task 2.

This module integrates:
- the Task 1 vehicle model,
- the generated reference path,
- the pure pursuit controller,
- and a goal-reaching condition.
"""

from __future__ import annotations

from pathlib import Path
import csv
import numpy as np
import yaml

from task1.vehicle_model import propagate_vehicle_state
from task2.path_generator import generate_reference_path, compute_path_arc_lengths
from task2.pure_pursuit import (
    find_nearest_path_index,
    find_lookahead_point,
    compute_pure_pursuit_steering_angle,
    compute_cross_track_error,
)


def load_config(config_path: str | Path) -> dict:
    """
    Load simulation parameters from a YAML configuration file.
    """
    with open(config_path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def run_single_task2_simulation(config: dict, lookahead_distance_override_m: float | None = None) -> dict:
    """
    Run a single Task 2 pure pursuit simulation.

    Parameters
    ----------
    config : dict
        Configuration dictionary.
    lookahead_distance_override_m : float | None
        Optional lookahead distance override for tuning studies.

    Returns
    -------
    dict
        Simulation results.
    """
    vehicle_speed_mps = float(config["vehicle"]["velocity"])
    wheelbase_m = float(config["vehicle"]["wheelbase"])

    time_step_s = float(config["simulation"]["time_step"])
    duration_s = float(config["simulation"]["duration"])
    initial_state = np.array(config["simulation"]["initial_state"], dtype=float)

    lookahead_distance_m = (
        float(lookahead_distance_override_m)
        if lookahead_distance_override_m is not None
        else float(config["controller"]["lookahead_distance"])
    )

    goal_tolerance_m = float(config["controller"]["goal_tolerance"])
    max_steering_angle_rad = np.deg2rad(float(config["controller"]["max_steering_deg"]))

    reference_path = generate_reference_path(
        x_start_m=float(config["path"]["x_start"]),
        x_end_m=float(config["path"]["x_end"]),
        num_points=int(config["path"]["num_points"]),
        amplitude_m=float(config["path"]["amplitude"]),
        wavelength_m=float(config["path"]["wavelength"]),
    )
    path_arc_lengths_m = compute_path_arc_lengths(reference_path)
    goal_point_xy_m = reference_path[-1]

    max_steps = int(np.floor(duration_s / time_step_s)) + 1

    time_history_s = []
    state_history = []
    steering_history_rad = []
    cross_track_error_history_m = []
    lookahead_point_history = []
    goal_distance_history_m = []

    current_state = initial_state.copy()
    goal_reached = False

    for step_index in range(max_steps):
        current_time_s = step_index * time_step_s
        current_position_xy_m = current_state[:2]

        nearest_index = find_nearest_path_index(reference_path, current_position_xy_m)
        lookahead_point_xy_m, _ = find_lookahead_point(
            reference_path=reference_path,
            path_arc_lengths_m=path_arc_lengths_m,
            nearest_index=nearest_index,
            lookahead_distance_m=lookahead_distance_m,
        )

        steering_angle_rad = compute_pure_pursuit_steering_angle(
            vehicle_state=current_state,
            lookahead_point_xy_m=lookahead_point_xy_m,
            wheelbase_m=wheelbase_m,
            lookahead_distance_m=lookahead_distance_m,
            max_steering_angle_rad=max_steering_angle_rad,
        )

        cross_track_error_m = compute_cross_track_error(current_position_xy_m, reference_path)
        goal_distance_m = float(np.linalg.norm(goal_point_xy_m - current_position_xy_m))

        time_history_s.append(current_time_s)
        state_history.append(current_state.copy())
        steering_history_rad.append(steering_angle_rad)
        cross_track_error_history_m.append(cross_track_error_m)
        lookahead_point_history.append(lookahead_point_xy_m.copy())
        goal_distance_history_m.append(goal_distance_m)

        if goal_distance_m <= goal_tolerance_m:
            goal_reached = True
            break

        current_state = propagate_vehicle_state(
            state=current_state,
            vehicle_speed_mps=vehicle_speed_mps,
            steering_angle_rad=steering_angle_rad,
            wheelbase_m=wheelbase_m,
            time_step_s=time_step_s,
        )

    state_history = np.array(state_history, dtype=float)
    lookahead_point_history = np.array(lookahead_point_history, dtype=float)

    return {
        "time_s": np.array(time_history_s, dtype=float),
        "state_history": state_history,
        "steering_history_rad": np.array(steering_history_rad, dtype=float),
        "cross_track_error_history_m": np.array(cross_track_error_history_m, dtype=float),
        "lookahead_point_history": lookahead_point_history,
        "goal_distance_history_m": np.array(goal_distance_history_m, dtype=float),
        "reference_path": reference_path,
        "goal_point_xy_m": goal_point_xy_m,
        "goal_reached": goal_reached,
        "lookahead_distance_m": lookahead_distance_m,
        "goal_tolerance_m": goal_tolerance_m,
        "vehicle_speed_mps": vehicle_speed_mps,
        "wheelbase_m": wheelbase_m,
        "time_step_s": time_step_s,
        "duration_s": duration_s,
    }


def run_lookahead_sweep(config: dict) -> list[dict]:
    """
    Run multiple Task 2 simulations for different lookahead distances.

    Parameters
    ----------
    config : dict
        Configuration dictionary.

    Returns
    -------
    list[dict]
        List of results dictionaries, one per lookahead value.
    """
    lookahead_values_m = [float(value) for value in config["analysis"]["lookahead_values"]]

    results_list = []
    for lookahead_distance_m in lookahead_values_m:
        results = run_single_task2_simulation(
            config=config,
            lookahead_distance_override_m=lookahead_distance_m
        )
        results_list.append(results)

    return results_list


def save_task2_results_to_csv(results: dict, output_csv_path: str | Path) -> None:
    """
    Save a single Task 2 simulation result to CSV.

    Parameters
    ----------
    results : dict
        Results dictionary.
    output_csv_path : str | Path
        Output CSV path.
    """
    output_csv_path = Path(output_csv_path)
    output_csv_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_csv_path, "w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow([
            "time_s",
            "x_m",
            "y_m",
            "heading_rad",
            "steering_angle_rad",
            "cross_track_error_m",
            "goal_distance_m",
            "lookahead_x_m",
            "lookahead_y_m",
        ])

        for index in range(len(results["time_s"])):
            writer.writerow([
                results["time_s"][index],
                results["state_history"][index, 0],
                results["state_history"][index, 1],
                results["state_history"][index, 2],
                results["steering_history_rad"][index],
                results["cross_track_error_history_m"][index],
                results["goal_distance_history_m"][index],
                results["lookahead_point_history"][index, 0],
                results["lookahead_point_history"][index, 1],
            ])