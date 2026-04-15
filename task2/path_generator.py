"""
path_generator.py

Reference path generation for ENGM020 Coursework 2, Part 2, Task 2.

This module creates a smooth 2D path for the pure pursuit controller to follow.
A sinusoidal path is used because it is smooth, easy to interpret, and suitable
for demonstrating the effect of controller tuning.
"""

from __future__ import annotations

import numpy as np


def generate_reference_path(
    x_start_m: float,
    x_end_m: float,
    num_points: int,
    amplitude_m: float,
    wavelength_m: float
) -> np.ndarray:
    """
    Generate a smooth sinusoidal reference path.

    Parameters
    ----------
    x_start_m : float
        Start x-position in metres.
    x_end_m : float
        End x-position in metres.
    num_points : int
        Number of path points to generate.
    amplitude_m : float
        Path amplitude in metres.
    wavelength_m : float
        Path wavelength in metres.

    Returns
    -------
    np.ndarray
        Path array of shape (N, 2), where each row is [x_m, y_m].
    """
    x_values_m = np.linspace(x_start_m, x_end_m, num_points)
    y_values_m = amplitude_m * np.sin(2.0 * np.pi * x_values_m / wavelength_m)

    return np.column_stack((x_values_m, y_values_m))


def compute_path_arc_lengths(reference_path: np.ndarray) -> np.ndarray:
    """
    Compute cumulative arc length along the reference path.

    Parameters
    ----------
    reference_path : np.ndarray
        Path array of shape (N, 2).

    Returns
    -------
    np.ndarray
        Cumulative arc length at each path point in metres.
    """
    segment_lengths_m = np.linalg.norm(np.diff(reference_path, axis=0), axis=1)
    cumulative_arc_length_m = np.concatenate(([0.0], np.cumsum(segment_lengths_m)))
    return cumulative_arc_length_m