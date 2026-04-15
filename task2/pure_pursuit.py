"""
pure_pursuit.py

Pure pursuit controller implementation for ENGM020 Coursework 2, Part 2, Task 2.
"""

from __future__ import annotations

import numpy as np


def find_nearest_path_index(reference_path: np.ndarray, vehicle_position_xy_m: np.ndarray) -> int:
    """
    Find the index of the nearest point on the reference path.

    Parameters
    ----------
    reference_path : np.ndarray
        Path array of shape (N, 2).
    vehicle_position_xy_m : np.ndarray
        Vehicle position as [x_m, y_m].

    Returns
    -------
    int
        Index of the nearest path point.
    """
    distances_m = np.linalg.norm(reference_path - vehicle_position_xy_m, axis=1)
    return int(np.argmin(distances_m))


def find_lookahead_point(
    reference_path: np.ndarray,
    path_arc_lengths_m: np.ndarray,
    nearest_index: int,
    lookahead_distance_m: float
) -> tuple[np.ndarray, int]:
    """
    Find the first path point at least the lookahead distance ahead of the nearest point.

    Parameters
    ----------
    reference_path : np.ndarray
        Path array of shape (N, 2).
    path_arc_lengths_m : np.ndarray
        Cumulative arc length values along the path.
    nearest_index : int
        Index of the nearest path point.
    lookahead_distance_m : float
        Lookahead distance in metres.

    Returns
    -------
    tuple[np.ndarray, int]
        Lookahead point [x_m, y_m] and its index.
    """
    target_arc_length_m = path_arc_lengths_m[nearest_index] + lookahead_distance_m

    for index in range(nearest_index, len(reference_path)):
        if path_arc_lengths_m[index] >= target_arc_length_m:
            return reference_path[index], index

    return reference_path[-1], len(reference_path) - 1


def transform_point_to_vehicle_frame(
    vehicle_state: np.ndarray,
    target_point_xy_m: np.ndarray
) -> np.ndarray:
    """
    Transform a global target point into the vehicle coordinate frame.

    Parameters
    ----------
    vehicle_state : np.ndarray
        Vehicle state [x_m, y_m, heading_rad].
    target_point_xy_m : np.ndarray
        Target point [x_m, y_m] in the global frame.

    Returns
    -------
    np.ndarray
        Target point [x_vehicle_m, y_vehicle_m] in the vehicle frame.
    """
    vehicle_x_m, vehicle_y_m, vehicle_heading_rad = vehicle_state

    dx_m = target_point_xy_m[0] - vehicle_x_m
    dy_m = target_point_xy_m[1] - vehicle_y_m

    x_vehicle_m = np.cos(vehicle_heading_rad) * dx_m + np.sin(vehicle_heading_rad) * dy_m
    y_vehicle_m = -np.sin(vehicle_heading_rad) * dx_m + np.cos(vehicle_heading_rad) * dy_m

    return np.array([x_vehicle_m, y_vehicle_m], dtype=float)


def compute_pure_pursuit_steering_angle(
    vehicle_state: np.ndarray,
    lookahead_point_xy_m: np.ndarray,
    wheelbase_m: float,
    lookahead_distance_m: float,
    max_steering_angle_rad: float
) -> float:
    """
    Compute steering angle using pure pursuit geometry.

    Parameters
    ----------
    vehicle_state : np.ndarray
        Vehicle state [x_m, y_m, heading_rad].
    lookahead_point_xy_m : np.ndarray
        Lookahead point [x_m, y_m] in the global frame.
    wheelbase_m : float
        Vehicle wheelbase in metres.
    lookahead_distance_m : float
        Lookahead distance in metres.
    max_steering_angle_rad : float
        Steering saturation limit in radians.

    Returns
    -------
    float
        Steering angle in radians.
    """
    lookahead_point_vehicle_frame = transform_point_to_vehicle_frame(
        vehicle_state=vehicle_state,
        target_point_xy_m=lookahead_point_xy_m
    )

    lateral_offset_m = lookahead_point_vehicle_frame[1]

    curvature_per_m = 2.0 * lateral_offset_m / (lookahead_distance_m ** 2)
    steering_angle_rad = np.arctan(wheelbase_m * curvature_per_m)

    steering_angle_rad = np.clip(
        steering_angle_rad,
        -max_steering_angle_rad,
        max_steering_angle_rad
    )

    return float(steering_angle_rad)


def compute_cross_track_error(
    vehicle_position_xy_m: np.ndarray,
    reference_path: np.ndarray
) -> float:
    """
    Compute the minimum Euclidean distance from the vehicle to the path.

    Parameters
    ----------
    vehicle_position_xy_m : np.ndarray
        Vehicle position [x_m, y_m].
    reference_path : np.ndarray
        Path array of shape (N, 2).

    Returns
    -------
    float
        Cross-track error in metres.
    """
    distances_m = np.linalg.norm(reference_path - vehicle_position_xy_m, axis=1)
    return float(np.min(distances_m))