"""
sensor_model.py

Sensor model for ENGM020 Coursework 2, Part 2, Task 1.

This module generates noisy GPS-style position measurements from the true
vehicle state. The measurements are corrupted by zero-mean Gaussian noise.
"""

from __future__ import annotations

import numpy as np


def generate_gps_measurement(
    true_state: np.ndarray,
    gps_noise_std_m: float,
    random_number_generator: np.random.Generator
) -> np.ndarray:
    """
    Generate a noisy GPS-style position measurement.

    Parameters
    ----------
    true_state : np.ndarray
        True vehicle state as [x_m, y_m, heading_rad].
    gps_noise_std_m : float
        Standard deviation of the GPS measurement noise in metres.
    random_number_generator : np.random.Generator
        NumPy random number generator for reproducibility.

    Returns
    -------
    np.ndarray
        Noisy position measurement as [x_meas_m, y_meas_m].
    """
    x_true_m, y_true_m, _ = true_state

    x_meas_m = x_true_m + random_number_generator.normal(0.0, gps_noise_std_m)
    y_meas_m = y_true_m + random_number_generator.normal(0.0, gps_noise_std_m)

    return np.array([x_meas_m, y_meas_m], dtype=float)