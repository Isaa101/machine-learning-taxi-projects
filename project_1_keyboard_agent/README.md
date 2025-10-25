# Project 1: Keyboard-Controlled Agent & Data Logging

This project implements a manually controlled agent in the Taxi-v3 environment using keyboard inputs. The goal is to generate a dataset of state-action pairs for future behavioral cloning experiments.

## Contents

- `keyboard_taxi.py`: Main script to run the Taxi-v3 environment with keyboard control (without storing data).
- `keyboard_taxi_logger.py`: Logs each step of the agent, storing state, action, reward, and other metadata.
- `map_1.txt`: Custom map configuration to fix passenger and destination positions.
- `map_loader.py`: Loads the custom map into the environment.
- `registry.py`: Registers the modified Taxi-v3 environment with Gymnasium.

## Features

- Keyboard control using `pynput` (arrow keys for movement, space for pickup, enter for dropoff).
- Logging of over 1000 experiences into a CSV file.
- Analysis of action distribution, state visitation, and agent behavior.

## How to Run

1. Install dependencies:
   ```bash
   pip install gymnasium pynput

2. Run the keyboard agent:
   ```bash
   python keyboar_taxi_logger.py
   
4. After playing several episodes, check experiences.csv for the logged data.

## Dataset
The logged experiences include:

- Encoded state
- Taxi position (row, col)
- Passenger location
- Destination index
- Action taken
- Reward received
- Episode termination flag

## Notes

The environment uses a fixed passenger and destination location to simplify analysis.
The is_raining flag is not supported in the current Gymnasium version.
This dataset is used in Project 2 for training a decision tree agent.
