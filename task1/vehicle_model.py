"""
vehicle_model.py

Vehicle kinematic model for ENGM020 Coursework 2, Task 1.

This module implements a simple bicycle-model / Dubins-style vehicle moving in
a 2D plane. The model is kinematic, meaning it describes how the vehicle pose
changes with steering and speed, without modelling forces or tyre dynamics.
"""

from __future__ import annotations

import numpy as np


def wrap_angle(angle_rad: float) -> float:
    """
    Wrap an angle to the interval [-pi, pi].

    Parameters
    ----------
    angle_rad : float
        Angle in radians.

    Returns
    -------
    float
        Wrapped angle in radians.
    """
    return (angle_rad + np.pi) % (2.0 * np.pi) - np.pi


def propagate_vehicle_state(
    state: np.ndarray,
    vehicle_speed_mps: float,
    steering_angle_rad: float,
    wheelbase_m: float,
    time_step_s: float
) -> np.ndarray:
    """
    Propagate the vehicle state forward by one discrete time step.

    The continuous-time model is:
        x_dot     = v cos(theta)
        y_dot     = v sin(theta)
        theta_dot = (v / L) tan(delta)

    A forward Euler method is used for the discrete-time update.

    Parameters
    ----------
    state : np.ndarray
        Current vehicle state as [x_m, y_m, heading_rad].
    vehicle_speed_mps : float
        Forward speed in metres per second.
    steering_angle_rad : float
        Steering angle in radians.
    wheelbase_m : float
        Vehicle wheelbase in metres.
    time_step_s : float
        Simulation time step in seconds.

    Returns
    -------
    np.ndarray
        Updated vehicle state as [x_m, y_m, heading_rad].
    """
    x_m, y_m, heading_rad = state

    x_dot_mps = vehicle_speed_mps * np.cos(heading_rad)
    y_dot_mps = vehicle_speed_mps * np.sin(heading_rad)
    heading_rate_radps = (vehicle_speed_mps / wheelbase_m) * np.tan(steering_angle_rad)

    x_next_m = x_m + x_dot_mps * time_step_s
    y_next_m = y_m + y_dot_mps * time_step_s
    heading_next_rad = wrap_angle(heading_rad + heading_rate_radps * time_step_s)

    return np.array([x_next_m, y_next_m, heading_next_rad], dtype=float)


def steering_profile(time_s: float) -> float:
    """
    Define the steering input profile used for Task 1 open-loop validation.

    This profile is deliberately simple so that the vehicle exhibits:
    - an initial straight segment,
    - a left turn,
    - a right turn,
    - and a final straight section.

    Parameters
    ----------
    time_s : float
        Current simulation time in seconds.

    Returns
    -------
    float
        Steering angle in radians.
    """
    if time_s < 3.0:
        steering_angle_deg = 0.0
    elif time_s < 6.0:
        steering_angle_deg = 10.0
    elif time_s < 9.0:
        steering_angle_deg = -8.0
    else:
        steering_angle_deg = 0.0

    return np.deg2rad(steering_angle_deg)