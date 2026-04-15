"""
kalman_filters.py

Kalman Filter and Extended Kalman Filter implementations for
ENGM020 Coursework 2, Part 2, Task 3.

IMPORTANT IMPLEMENTATION NOTE
-----------------------------
The supplied source file `kalman_filter.py` was used as the starting reference
for the estimator structure. However, inspection of the provided code showed
that its "KF" prediction step already used state-dependent nonlinear terms
cos(theta), sin(theta), tan(delta), together with a state-dependent Jacobian-
like A matrix. For the present vehicle model, that makes the supplied KF
formulation effectively EKF-like.

To obtain a meaningful KF versus EKF comparison for the coursework, the
following approach is used here:

1. KF baseline:
   A true linear Kalman Filter is implemented using a small-angle linearisation
   of the bicycle model about theta ≈ 0 and delta ≈ 0.
   This gives a genuine linear baseline.

2. EKF:
   The EKF uses the full nonlinear bicycle-model state transition and the
   corresponding motion Jacobian.

This keeps the supplied estimator idea as the project starting point, but
corrects the formulation so that the KF and EKF are mathematically distinct
and therefore comparable.
"""

from __future__ import annotations

import numpy as np

from task1.vehicle_model import wrap_angle


def build_filter_matrices(config: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build the Q, R, and P0 covariance matrices from the configuration file.

    Parameters
    ----------
    config : dict
        Project configuration dictionary.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray]
        Process noise covariance Q, measurement noise covariance R,
        and initial covariance P0.
    """
    process_noise_position = float(config["kalman_filter"]["process_noise_position"])
    process_noise_heading_rad = np.deg2rad(
        float(config["kalman_filter"]["process_noise_heading_deg"])
    )

    measurement_noise_position = float(config["kalman_filter"]["measurement_noise_position"])
    measurement_noise_heading_rad = np.deg2rad(
        float(config["kalman_filter"]["measurement_noise_heading_deg"])
    )

    initial_position_std = float(config["kalman_filter"]["initial_position_std"])
    initial_heading_std_rad = np.deg2rad(
        float(config["kalman_filter"]["initial_heading_std_deg"])
    )

    Q = np.diag([
        process_noise_position**2,
        process_noise_position**2,
        process_noise_heading_rad**2,
    ])

    R = np.diag([
        measurement_noise_position**2,
        measurement_noise_position**2,
        measurement_noise_heading_rad**2,
    ])

    P0 = np.diag([
        initial_position_std**2,
        initial_position_std**2,
        initial_heading_std_rad**2,
    ])

    return Q, R, P0


def linear_kf_step(
    x_est: np.ndarray,
    P: np.ndarray,
    control_input: np.ndarray,
    measurement: np.ndarray,
    time_step_s: float,
    wheelbase_m: float,
    Q: np.ndarray,
    R: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """
    Perform one step of a true linear Kalman Filter baseline.

    CHANGE FROM SUPPLIED KF:
    ------------------------
    The supplied source file used theta-dependent cos(theta), sin(theta),
    tan(delta) terms directly in the prediction step. That makes the supplied
    "KF" effectively a locally linearised nonlinear filter.

    Here, the KF is reformulated as a genuine linear baseline using a
    small-angle linearisation of the bicycle model about:

        theta ≈ 0
        delta ≈ 0

    Starting from the nonlinear model:
        x_{k+1} = x_k + v cos(theta_k) dt
        y_{k+1} = y_k + v sin(theta_k) dt
        theta_{k+1} = theta_k + (v/L) tan(delta_k) dt

    Small-angle approximations:
        cos(theta) ≈ 1
        sin(theta) ≈ theta
        tan(delta) ≈ delta

    give the linear baseline model:
        x_{k+1}     ≈ x_k + v dt
        y_{k+1}     ≈ y_k + v theta_k dt
        theta_{k+1} ≈ theta_k + (v/L) delta_k dt

    This produces a mathematically distinct linear KF, which can be compared
    fairly against the EKF.

    Parameters
    ----------
    x_est : np.ndarray
        Current state estimate [x_m, y_m, heading_rad].
    P : np.ndarray
        Current covariance matrix.
    control_input : np.ndarray
        Control input [vehicle_speed_mps, steering_angle_rad].
    measurement : np.ndarray
        Measurement vector [x_meas_m, y_meas_m, heading_meas_rad].
    time_step_s : float
        Simulation time step in seconds.
    wheelbase_m : float
        Vehicle wheelbase in metres.
    Q : np.ndarray
        Process noise covariance matrix.
    R : np.ndarray
        Measurement noise covariance matrix.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Updated state estimate and covariance matrix.
    """
    vehicle_speed_mps, steering_angle_rad = control_input

    # True linear KF baseline: small-angle linearisation about theta ≈ 0.
    A = np.array([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, vehicle_speed_mps * time_step_s],
        [0.0, 0.0, 1.0],
    ])

    # Affine known-input term for the linearised model.
    # tan(delta) ≈ delta is used here to keep the KF genuinely linearised.
    u_affine = np.array([
        vehicle_speed_mps * time_step_s,
        0.0,
        (vehicle_speed_mps / wheelbase_m) * steering_angle_rad * time_step_s,
    ])

    # Direct state measurement model:
    # z = H x + v, with H = I
    H = np.eye(3)

    # Predict
    x_pred = A @ x_est + u_affine
    x_pred[2] = wrap_angle(x_pred[2])

    P_pred = A @ P @ A.T + Q

    # Correct
    innovation = measurement - H @ x_pred
    innovation[2] = wrap_angle(innovation[2])

    S = H @ P_pred @ H.T + R
    K = P_pred @ H.T @ np.linalg.inv(S)

    x_updated = x_pred + K @ innovation
    x_updated[2] = wrap_angle(x_updated[2])

    P_updated = (np.eye(3) - K @ H) @ P_pred

    return x_updated, P_updated


def nonlinear_state_transition(
    state: np.ndarray,
    control_input: np.ndarray,
    time_step_s: float,
    wheelbase_m: float
) -> np.ndarray:
    """
    Nonlinear bicycle-model state transition for the EKF.

    EKF-SPECIFIC CHANGE:
    --------------------
    Unlike the linear KF baseline, the EKF keeps the full nonlinear bicycle
    model in the prediction step.

    Parameters
    ----------
    state : np.ndarray
        State vector [x_m, y_m, heading_rad].
    control_input : np.ndarray
        Control input [vehicle_speed_mps, steering_angle_rad].
    time_step_s : float
        Time step in seconds.
    wheelbase_m : float
        Vehicle wheelbase in metres.

    Returns
    -------
    np.ndarray
        Predicted next state.
    """
    x_m, y_m, heading_rad = state
    vehicle_speed_mps, steering_angle_rad = control_input

    x_next_m = x_m + vehicle_speed_mps * np.cos(heading_rad) * time_step_s
    y_next_m = y_m + vehicle_speed_mps * np.sin(heading_rad) * time_step_s
    heading_next_rad = wrap_angle(
        heading_rad + (vehicle_speed_mps / wheelbase_m) * np.tan(steering_angle_rad) * time_step_s
    )

    return np.array([x_next_m, y_next_m, heading_next_rad], dtype=float)


def compute_motion_jacobian(
    state: np.ndarray,
    control_input: np.ndarray,
    time_step_s: float
) -> np.ndarray:
    """
    Compute the Jacobian of the nonlinear motion model with respect to state.

    EKF-SPECIFIC CHANGE:
    --------------------
    The EKF replaces the fixed linear state transition matrix with the Jacobian
    of the nonlinear bicycle model evaluated about the current estimate.

    Parameters
    ----------
    state : np.ndarray
        Current state [x_m, y_m, heading_rad].
    control_input : np.ndarray
        Control input [vehicle_speed_mps, steering_angle_rad].
    time_step_s : float
        Time step in seconds.

    Returns
    -------
    np.ndarray
        Motion model Jacobian.
    """
    heading_rad = state[2]
    vehicle_speed_mps = control_input[0]

    F = np.array([
        [1.0, 0.0, -vehicle_speed_mps * np.sin(heading_rad) * time_step_s],
        [0.0, 1.0,  vehicle_speed_mps * np.cos(heading_rad) * time_step_s],
        [0.0, 0.0,  1.0],
    ])

    return F


def ekf_step(
    x_est: np.ndarray,
    P: np.ndarray,
    control_input: np.ndarray,
    measurement: np.ndarray,
    time_step_s: float,
    wheelbase_m: float,
    Q: np.ndarray,
    R: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """
    Perform one step of the Extended Kalman Filter.

    EKF IMPLEMENTATION SUMMARY:
    ---------------------------
    1. Predict using the full nonlinear bicycle model f(x, u).
    2. Compute the Jacobian F = ∂f/∂x about the current estimate.
    3. Propagate covariance with F instead of a fixed linear A matrix.
    4. Correct using the same direct measurement model z = Hx + v.

    Parameters
    ----------
    x_est : np.ndarray
        Current state estimate [x_m, y_m, heading_rad].
    P : np.ndarray
        Current covariance matrix.
    control_input : np.ndarray
        Control input [vehicle_speed_mps, steering_angle_rad].
    measurement : np.ndarray
        Measurement vector [x_meas_m, y_meas_m, heading_meas_rad].
    time_step_s : float
        Simulation time step in seconds.
    wheelbase_m : float
        Vehicle wheelbase in metres.
    Q : np.ndarray
        Process noise covariance matrix.
    R : np.ndarray
        Measurement noise covariance matrix.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Updated state estimate and covariance matrix.
    """
    # Direct state measurement model.
    H = np.eye(3)

    # Predict with nonlinear model
    x_pred = nonlinear_state_transition(
        state=x_est,
        control_input=control_input,
        time_step_s=time_step_s,
        wheelbase_m=wheelbase_m,
    )

    # Linearise about the current estimate for covariance propagation
    F = compute_motion_jacobian(
        state=x_est,
        control_input=control_input,
        time_step_s=time_step_s,
    )

    P_pred = F @ P @ F.T + Q

    # Correct
    innovation = measurement - H @ x_pred
    innovation[2] = wrap_angle(innovation[2])

    S = H @ P_pred @ H.T + R
    K = P_pred @ H.T @ np.linalg.inv(S)

    x_updated = x_pred + K @ innovation
    x_updated[2] = wrap_angle(x_updated[2])

    P_updated = (np.eye(3) - K @ H) @ P_pred

    return x_updated, P_updated