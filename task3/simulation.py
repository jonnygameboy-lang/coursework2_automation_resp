"""
simulation.py

Task 3 simulation for ENGM020 Coursework 2, Part 2.

This module integrates:
- Task 2 pure pursuit path tracking,
- Task 3 noisy state measurements,
- a baseline KF,
- and an EKF,
so that the estimators can be compared on the same closed-loop scenario.
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
)
from task3.measurement_model import generate_noisy_state_measurement
from task3.kalman_filters import build_filter_matrices, linear_kf_step, ekf_step


def load_config(config_path: str | Path) -> dict:
    """
    Load project configuration from YAML.
    """
    with open(config_path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def compute_state_error(true_states: np.ndarray, estimated_states: np.ndarray) -> np.ndarray:
    """
    Compute estimation error history.

    Parameters
    ----------
    true_states : np.ndarray
        True state history of shape (N, 3).
    estimated_states : np.ndarray
        Estimated state history of shape (N, 3).

    Returns
    -------
    np.ndarray
        Error history of shape (N, 3).
    """
    errors = estimated_states - true_states
    errors[:, 2] = (errors[:, 2] + np.pi) % (2.0 * np.pi) - np.pi
    return errors


def run_task3_simulation(config: dict) -> dict:
    """
    Run the full Task 3 estimator comparison simulation.

    Parameters
    ----------
    config : dict
        Configuration dictionary.

    Returns
    -------
    dict
        Results dictionary containing true state, noisy measurements,
        KF estimates, EKF estimates, and performance metrics.
    """
    vehicle_speed_mps = float(config["vehicle"]["velocity"])
    wheelbase_m = float(config["vehicle"]["wheelbase"])

    lookahead_distance_m = float(config["controller"]["lookahead_distance"])
    goal_tolerance_m = float(config["controller"]["goal_tolerance"])
    max_steering_angle_rad = np.deg2rad(float(config["controller"]["max_steering_deg"]))

    time_step_s = float(config["simulation"]["time_step"])
    duration_s = float(config["simulation"]["duration"])
    initial_state = np.array(config["simulation"]["initial_state"], dtype=float)
    random_seed = int(config["simulation"]["random_seed"])

    position_noise_std_m = float(config["sensor"]["gps_noise_std"])
    heading_noise_std_rad = np.deg2rad(float(config["sensor"]["heading_noise_std_deg"]))

    reference_path = generate_reference_path(
        x_start_m=float(config["path"]["x_start"]),
        x_end_m=float(config["path"]["x_end"]),
        num_points=int(config["path"]["num_points"]),
        amplitude_m=float(config["path"]["amplitude"]),
        wavelength_m=float(config["path"]["wavelength"]),
    )
    path_arc_lengths_m = compute_path_arc_lengths(reference_path)
    goal_point_xy_m = reference_path[-1]

    random_number_generator = np.random.default_rng(random_seed)
    Q, R, P0 = build_filter_matrices(config)

    max_steps = int(np.floor(duration_s / time_step_s)) + 1

    time_history_s = []
    true_state_history = []
    measurement_history = []
    steering_history_rad = []
    kf_state_history = []
    ekf_state_history = []
    goal_distance_history_m = []

    true_state = initial_state.copy()
    first_measurement = generate_noisy_state_measurement(
        true_state=true_state,
        position_noise_std_m=position_noise_std_m,
        heading_noise_std_rad=heading_noise_std_rad,
        random_number_generator=random_number_generator,
    )

    kf_state_estimate = first_measurement.copy()
    ekf_state_estimate = first_measurement.copy()
    kf_covariance = P0.copy()
    ekf_covariance = P0.copy()

    goal_reached = False

    for step_index in range(max_steps):
        current_time_s = step_index * time_step_s
        current_position_xy_m = true_state[:2]

        nearest_index = find_nearest_path_index(reference_path, current_position_xy_m)
        lookahead_point_xy_m, _ = find_lookahead_point(
            reference_path=reference_path,
            path_arc_lengths_m=path_arc_lengths_m,
            nearest_index=nearest_index,
            lookahead_distance_m=lookahead_distance_m,
        )

        steering_angle_rad = compute_pure_pursuit_steering_angle(
            vehicle_state=true_state,
            lookahead_point_xy_m=lookahead_point_xy_m,
            wheelbase_m=wheelbase_m,
            lookahead_distance_m=lookahead_distance_m,
            max_steering_angle_rad=max_steering_angle_rad,
        )

        control_input = np.array([vehicle_speed_mps, steering_angle_rad], dtype=float)

        measurement = generate_noisy_state_measurement(
            true_state=true_state,
            position_noise_std_m=position_noise_std_m,
            heading_noise_std_rad=heading_noise_std_rad,
            random_number_generator=random_number_generator,
        )

        kf_state_estimate, kf_covariance = linear_kf_step(
            x_est=kf_state_estimate,
            P=kf_covariance,
            control_input=control_input,
            measurement=measurement,
            time_step_s=time_step_s,
            wheelbase_m=wheelbase_m,
            Q=Q,
            R=R,
        )

        ekf_state_estimate, ekf_covariance = ekf_step(
            x_est=ekf_state_estimate,
            P=ekf_covariance,
            control_input=control_input,
            measurement=measurement,
            time_step_s=time_step_s,
            wheelbase_m=wheelbase_m,
            Q=Q,
            R=R,
        )

        goal_distance_m = float(np.linalg.norm(goal_point_xy_m - current_position_xy_m))

        time_history_s.append(current_time_s)
        true_state_history.append(true_state.copy())
        measurement_history.append(measurement.copy())
        steering_history_rad.append(steering_angle_rad)
        kf_state_history.append(kf_state_estimate.copy())
        ekf_state_history.append(ekf_state_estimate.copy())
        goal_distance_history_m.append(goal_distance_m)

        if goal_distance_m <= goal_tolerance_m:
            goal_reached = True
            break

        true_state = propagate_vehicle_state(
            state=true_state,
            vehicle_speed_mps=vehicle_speed_mps,
            steering_angle_rad=steering_angle_rad,
            wheelbase_m=wheelbase_m,
            time_step_s=time_step_s,
        )

    time_history_s = np.array(time_history_s, dtype=float)
    true_state_history = np.array(true_state_history, dtype=float)
    measurement_history = np.array(measurement_history, dtype=float)
    steering_history_rad = np.array(steering_history_rad, dtype=float)
    kf_state_history = np.array(kf_state_history, dtype=float)
    ekf_state_history = np.array(ekf_state_history, dtype=float)
    goal_distance_history_m = np.array(goal_distance_history_m, dtype=float)

    measurement_errors = compute_state_error(true_state_history, measurement_history)
    kf_errors = compute_state_error(true_state_history, kf_state_history)
    ekf_errors = compute_state_error(true_state_history, ekf_state_history)

    return {
        "time_s": time_history_s,
        "reference_path": reference_path,
        "goal_point_xy_m": goal_point_xy_m,
        "goal_reached": goal_reached,
        "true_state_history": true_state_history,
        "measurement_history": measurement_history,
        "steering_history_rad": steering_history_rad,
        "kf_state_history": kf_state_history,
        "ekf_state_history": ekf_state_history,
        "measurement_errors": measurement_errors,
        "kf_errors": kf_errors,
        "ekf_errors": ekf_errors,
        "goal_distance_history_m": goal_distance_history_m,
        "lookahead_distance_m": lookahead_distance_m,
        "time_step_s": time_step_s,
        "wheelbase_m": wheelbase_m,
        "vehicle_speed_mps": vehicle_speed_mps,
    }


def save_task3_results_to_csv(results: dict, output_csv_path: str | Path) -> None:
    """
    Save Task 3 results to CSV.

    Parameters
    ----------
    results : dict
        Results dictionary from the simulation.
    output_csv_path : str | Path
        Output CSV file path.
    """
    output_csv_path = Path(output_csv_path)
    output_csv_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_csv_path, "w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow([
            "time_s",
            "x_true_m", "y_true_m", "heading_true_rad",
            "x_meas_m", "y_meas_m", "heading_meas_rad",
            "x_kf_m", "y_kf_m", "heading_kf_rad",
            "x_ekf_m", "y_ekf_m", "heading_ekf_rad",
            "steering_angle_rad",
            "goal_distance_m",
        ])

        for index in range(len(results["time_s"])):
            writer.writerow([
                results["time_s"][index],
                results["true_state_history"][index, 0],
                results["true_state_history"][index, 1],
                results["true_state_history"][index, 2],
                results["measurement_history"][index, 0],
                results["measurement_history"][index, 1],
                results["measurement_history"][index, 2],
                results["kf_state_history"][index, 0],
                results["kf_state_history"][index, 1],
                results["kf_state_history"][index, 2],
                results["ekf_state_history"][index, 0],
                results["ekf_state_history"][index, 1],
                results["ekf_state_history"][index, 2],
                results["steering_history_rad"][index],
                results["goal_distance_history_m"][index],
            ])