"""
run_task1.py

Main execution script for ENGM020 Coursework 2, Part 2, Task 1.

This script:
1. Loads the simulation configuration,
2. Runs the vehicle and sensor simulation,
3. Saves the numeric results to CSV,
4. Generates figures for the presentation slides.
"""

from __future__ import annotations

from pathlib import Path

from task1.simulation import (
    load_config,
    run_task1_simulation,
    save_simulation_results_to_csv,
)
from task1.plotting import create_task1_figures


def main() -> None:
    """
    Run the complete Task 1 workflow.
    """
    project_root = Path(__file__).resolve().parents[1]

    config_path = project_root / "config.yaml"
    figures_directory = project_root / "outputs" / "figures"
    data_directory = project_root / "outputs" / "data"
    output_csv_path = data_directory / "task1_simulation_results.csv"

    config = load_config(config_path)
    results = run_task1_simulation(config)

    save_simulation_results_to_csv(results, output_csv_path)
    create_task1_figures(results, figures_directory)

    print("Task 1 simulation completed successfully.")
    print(f"Results saved to: {output_csv_path}")
    print(f"Figures saved to: {figures_directory}")


if __name__ == "__main__":
    main()