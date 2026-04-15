"""
simulation.py

Simulation engine for ENGM020 Coursework 2, Part 2, Task 1.

This module runs the vehicle kinematic model together with the noisy sensor
model and stores the results in a structured dictionary.
"""

from __future__ import annotations

from pathlib import Path
import csv
import numpy as np
import yaml

from task1.vehicle_model import propagate_vehicle_state, steering_profile
from task1.sensor_model import generate_gps_measurement


def load_config(config_path: str | Path) -> dict:
    """
    Load simulation parameters from a YAML configuration file.

    Parameters
    ----------
    config_path : str | Path
        Path to the YAML configuration file.

    Returns
    -------
    dict
        Configuration dictionary.
    """
    with open(config_path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def run_task1_simulation(config: dict) -> dict:
    """
    Run the Task 1 vehicle and sensor simulation.

    Parameters
    ----------
    config : dict
        Configuration dictionary loaded from config.yaml.

    Returns
    -------
    dict
        Dictionary containing time history, true states, measurements,
        steering inputs, and parameters used in the simulation.
    """
    vehicle_speed_mps = float(config["vehicle"]["velocity"])
    wheelbase_m = float(config["vehicle"]["wheelbase"])
    gps_noise_std_m = float(config["sensor"]["gps_noise_std"])

    time_step_s = float(config["simulation"]["time_step"])
    duration_s = float(config["simulation"]["duration"])
    random_seed = int(config["simulation"]["random_seed"])
    initial_state = np.array(config["simulation"]["initial_state"], dtype=float)

    random_number_generator = np.random.default_rng(random_seed)

    time_vector_s = np.arange(0.0, duration_s + time_step_s, time_step_s)

    num_steps = len(time_vector_s)
    true_states = np.zeros((num_steps, 3), dtype=float)
    gps_measurements = np.zeros((num_steps, 2), dtype=float)
    steering_angles_rad = np.zeros(num_steps, dtype=float)

    true_states[0, :] = initial_state
    gps_measurements[0, :] = generate_gps_measurement(
        true_state=initial_state,
        gps_noise_std_m=gps_noise_std_m,
        random_number_generator=random_number_generator
    )
    steering_angles_rad[0] = steering_profile(time_vector_s[0])

    for step_index in range(1, num_steps):
        current_time_s = time_vector_s[step_index - 1]
        steering_angle_rad = steering_profile(current_time_s)

        previous_state = true_states[step_index - 1, :]
        updated_state = propagate_vehicle_state(
            state=previous_state,
            vehicle_speed_mps=vehicle_speed_mps,
            steering_angle_rad=steering_angle_rad,
            wheelbase_m=wheelbase_m,
            time_step_s=time_step_s
        )

        measurement = generate_gps_measurement(
            true_state=updated_state,
            gps_noise_std_m=gps_noise_std_m,
            random_number_generator=random_number_generator
        )

        true_states[step_index, :] = updated_state
        gps_measurements[step_index, :] = measurement
        steering_angles_rad[step_index] = steering_angle_rad

    return {
        "time_s": time_vector_s,
        "true_states": true_states,
        "gps_measurements": gps_measurements,
        "steering_angles_rad": steering_angles_rad,
        "vehicle_speed_mps": vehicle_speed_mps,
        "wheelbase_m": wheelbase_m,
        "gps_noise_std_m": gps_noise_std_m,
        "time_step_s": time_step_s,
        "duration_s": duration_s,
        "random_seed": random_seed,
    }


def save_simulation_results_to_csv(results: dict, output_csv_path: str | Path) -> None:
    """
    Save simulation results to a CSV file.

    Parameters
    ----------
    results : dict
        Simulation results dictionary.
    output_csv_path : str | Path
        Path to the output CSV file.
    """
    output_csv_path = Path(output_csv_path)
    output_csv_path.parent.mkdir(parents=True, exist_ok=True)

    time_s = results["time_s"]
    true_states = results["true_states"]
    gps_measurements = results["gps_measurements"]
    steering_angles_rad = results["steering_angles_rad"]

    with open(output_csv_path, "w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow([
            "time_s",
            "x_true_m",
            "y_true_m",
            "heading_true_rad",
            "x_gps_m",
            "y_gps_m",
            "steering_angle_rad",
        ])

        for index in range(len(time_s)):
            writer.writerow([
                time_s[index],
                true_states[index, 0],
                true_states[index, 1],
                true_states[index, 2],
                gps_measurements[index, 0],
                gps_measurements[index, 1],
                steering_angles_rad[index],
            ])