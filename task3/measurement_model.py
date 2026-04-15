"""
measurement_model.py

Measurement model for ENGM020 Coursework 2, Part 2, Task 3.

This module generates noisy measurements of vehicle position and heading for use
by the Kalman Filter (KF) and Extended Kalman Filter (EKF).
"""

from __future__ import annotations

import numpy as np

from task1.vehicle_model import wrap_angle


def generate_noisy_state_measurement(
    true_state: np.ndarray,
    position_noise_std_m: float,
    heading_noise_std_rad: float,
    random_number_generator: np.random.Generator
) -> np.ndarray:
    """
    Generate a noisy measurement of [x, y, theta].

    Parameters
    ----------
    true_state : np.ndarray
        True vehicle state [x_m, y_m, heading_rad].
    position_noise_std_m : float
        Standard deviation of x and y measurement noise in metres.
    heading_noise_std_rad : float
        Standard deviation of heading measurement noise in radians.
    random_number_generator : np.random.Generator
        Random number generator for reproducible results.

    Returns
    -------
    np.ndarray
        Noisy measurement vector [x_meas_m, y_meas_m, heading_meas_rad].
    """
    x_true_m, y_true_m, heading_true_rad = true_state

    x_meas_m = x_true_m + random_number_generator.normal(0.0, position_noise_std_m)
    y_meas_m = y_true_m + random_number_generator.normal(0.0, position_noise_std_m)
    heading_meas_rad = wrap_angle(
        heading_true_rad + random_number_generator.normal(0.0, heading_noise_std_rad)
    )

    return np.array([x_meas_m, y_meas_m, heading_meas_rad], dtype=float)