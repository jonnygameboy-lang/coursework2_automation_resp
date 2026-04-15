# ENGM020 Coursework 2 - Part 2 Automation Lab

# Requirements

Tasks 1–3 are intended to be run in Python from the project root.

Required Python packages:

numpy
matplotlib
pyyaml

Install dependencies with:

pip install -r requirements.txt

Shared parameters are stored in config.yaml. The main parameters that can be changed are:

vehicle.velocity
vehicle.wheelbase
sensor.gps_noise_std
controller.lookahead_distance
kalman_filter.process_noise
kalman_filter.measurement_noise

Additional fields may also be present in the final version of the config file for simulation timing, path shape, goal tolerance, heading noise, and estimator initial conditions.

# Running the tasks

Run all commands below from the project root.

Task 1 – Vehicle and sensor simulation

python -m task1.run_task1

This generates the Task 1 simulation outputs and figures.

Task 2 – Pure pursuit controller

python -m task2.run_task2

This generates the Task 2 path-tracking results, lookahead comparison, and figures.

Task 3 – KF / EKF estimation

python -m task3.run_task3

This generates the Task 3 estimator comparison results and figures.

Outputs

Generated figures are saved in:

outputs/figures/

Generated data files are saved in:

outputs/data/

Task 4 – ROS implementation

Task 4 is intended to be run as a ROS implementation rather than through the standalone Python task runners.

It uses:

an estimator node
a comparison / visualisation node

The intended workflow is:

launch the ROS/Gazebo environment
start the repeatable motion input
run the estimator node
run the comparison / visualisation node
inspect topics, node graph, and RViz output
review any logged CSV results

Task 4 assumes the user is already familiar with:

Linux terminal usage
ROS workspace setup
sourcing the correct environment
running nodes with rosrun
using Gazebo, RViz, rostopic, and rqt_graph

# Notes

The full source code, figures, and supporting files are included in the submitted archive and GitHub repository.