# UCSD ECE276B PR3 

## Overview
In this assignment, you will implement a controller for a car robot to track a trajectory.

## Dependencies
This starter code was tested with: python 3.7, matplotlib 3.4, and numpy 1.20. 

## Starter code
### 1. main.py
This file contains examples of how to generate control inputs from a simple P controller and apply the control on a car model. This simple controller does not work well. Your task is to replace the P controller with your own controller using CEC and GPI as described in the project statement.

### 2. utils.py
This file contains code to visualize the desired trajectory, robot's trajectory, and obstacles.


## My Implementation
The gpi_pytorch.py runs the value iteration. The P matrix will be saved after running python gpi_pytorch.py. The main_gpi will use the P matrix saved to do the simulation. 
The main_cec.py solves a non linear optimization problem for tracking a directory.

