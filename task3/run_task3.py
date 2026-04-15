"""
run_task3.py

Main execution script for ENGM020 Coursework 2, Part 2, Task 3.
"""

from __future__ import annotations

from pathlib import Path
import numpy as np

from task3.simulation import load_config, run_task3_simulation, save_task3_results_to_csv
from task3.plotting import create_task3_figures


def summarise_results(results: dict) -> None:
    """
    Print a concise numerical summary for Task 3.
    """
    measurement_position_error = np.linalg.norm(results["measurement_errors"][:, :2], axis=1)
    kf_position_error = np.linalg.norm(results["kf_errors"][:, :2], axis=1)
    ekf_position_error = np.linalg.norm(results["ekf_errors"][:, :2], axis=1)

    measurement_heading_error_deg = np.abs(np.rad2deg(results["measurement_errors"][:, 2]))
    kf_heading_error_deg = np.abs(np.rad2deg(results["kf_errors"][:, 2]))
    ekf_heading_error_deg = np.abs(np.rad2deg(results["ekf_errors"][:, 2]))

    print("-" * 60)
    print(f"Goal reached: {results['goal_reached']}")
    print(f"Simulation duration used: {results['time_s'][-1]:.2f} s")
    print()
    print("Position error metrics:")
    print(f"  Measurement mean: {np.mean(measurement_position_error):.3f} m")
    print(f"  KF mean:          {np.mean(kf_position_error):.3f} m")
    print(f"  EKF mean:         {np.mean(ekf_position_error):.3f} m")
    print(f"  Measurement RMSE: {np.sqrt(np.mean(measurement_position_error**2)):.3f} m")
    print(f"  KF RMSE:          {np.sqrt(np.mean(kf_position_error**2)):.3f} m")
    print(f"  EKF RMSE:         {np.sqrt(np.mean(ekf_position_error**2)):.3f} m")
    print()
    print("Heading error metrics:")
    print(f"  Measurement mean abs: {np.mean(measurement_heading_error_deg):.3f} deg")
    print(f"  KF mean abs:          {np.mean(kf_heading_error_deg):.3f} deg")
    print(f"  EKF mean abs:         {np.mean(ekf_heading_error_deg):.3f} deg")


def main() -> None:
    """
    Run the complete Task 3 workflow.
    """
    project_root = Path(__file__).resolve().parents[1]

    config_path = project_root / "config.yaml"
    figures_directory = project_root / "outputs" / "figures"
    data_directory = project_root / "outputs" / "data"
    output_csv_path = data_directory / "task3_estimator_results.csv"

    config = load_config(config_path)
    results = run_task3_simulation(config)

    save_task3_results_to_csv(results, output_csv_path)
    create_task3_figures(results, figures_directory)
    summarise_results(results)

    print("\nTask 3 outputs saved successfully.")
    print(f"Data saved to: {output_csv_path}")
    print(f"Figures saved to: {figures_directory}")


if __name__ == "__main__":
    main()