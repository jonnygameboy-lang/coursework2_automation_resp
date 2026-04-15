"""
run_task2.py

Main execution script for ENGM020 Coursework 2, Part 2, Task 2.

This script:
1. Loads the configuration,
2. Runs a baseline pure pursuit simulation,
3. Saves baseline data,
4. Generates baseline figures,
5. Runs a lookahead sweep,
6. Generates comparison figures,
7. Prints summary performance information.
"""

from __future__ import annotations

from pathlib import Path
import numpy as np

from task2.simulation import (
    load_config,
    run_single_task2_simulation,
    run_lookahead_sweep,
    save_task2_results_to_csv,
)
from task2.plotting import (
    create_single_run_figures,
    create_lookahead_comparison_figures,
)


def print_simulation_summary(results: dict) -> None:
    """
    Print a concise summary of a Task 2 run.
    """
    mean_cross_track_error_m = float(np.mean(results["cross_track_error_history_m"]))
    max_cross_track_error_m = float(np.max(results["cross_track_error_history_m"]))
    final_goal_distance_m = float(results["goal_distance_history_m"][-1])

    print("-" * 60)
    print(f"Lookahead distance: {results['lookahead_distance_m']:.2f} m")
    print(f"Goal reached: {results['goal_reached']}")
    print(f"Simulation duration used: {results['time_s'][-1]:.2f} s")
    print(f"Mean cross-track error: {mean_cross_track_error_m:.3f} m")
    print(f"Maximum cross-track error: {max_cross_track_error_m:.3f} m")
    print(f"Final distance to goal: {final_goal_distance_m:.3f} m")


def main() -> None:
    """
    Run the complete Task 2 workflow.
    """
    project_root = Path(__file__).resolve().parents[1]

    config_path = project_root / "config.yaml"
    figures_directory = project_root / "outputs" / "figures"
    data_directory = project_root / "outputs" / "data"
    baseline_csv_path = data_directory / "task2_baseline_results.csv"

    config = load_config(config_path)

    baseline_results = run_single_task2_simulation(config)
    save_task2_results_to_csv(baseline_results, baseline_csv_path)
    create_single_run_figures(baseline_results, figures_directory)

    print("Baseline Task 2 run completed.")
    print_simulation_summary(baseline_results)

    lookahead_results_list = run_lookahead_sweep(config)
    create_lookahead_comparison_figures(lookahead_results_list, figures_directory)

    print("\nLookahead sweep completed.")
    for results in lookahead_results_list:
        print_simulation_summary(results)

    print("\nTask 2 outputs saved successfully.")
    print(f"Data saved to: {baseline_csv_path}")
    print(f"Figures saved to: {figures_directory}")


if __name__ == "__main__":
    main()