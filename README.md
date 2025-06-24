# Project Overview
This project was undertaken during my internship at DRDO and focuses on designing an AI-based Predictive Maintenance System for critical mechanical and electronic components. The primary goal is to prevent unexpected failures by using machine learning to detect early warning signs based on system health parameters like temperature, vibrations, and usage time.

This system simulates a real-time health monitoring pipeline that analyzes operational data and predicts whether a component is likely to fail, helping improve reliability and reduce maintenance costs in mission-critical environments.

# Why This Project?
Conventional maintenance strategies like corrective or even preventive maintenance often lead to unplanned downtimes, especially in sensitive applications such as defense and aerospace.

The need for a predictive approach becomes crucial—where the system continuously monitors key indicators and forecasts failures before they occur. This ensures:

1.Minimal downtime

2.Efficient resource planning

3.Safer operations of mechanical systems

# What This Project Achieves
The main aim is to build a modular and extensible backend system that uses machine learning for:

1. Failure Prediction: Identify if a component is heading toward failure.

2. Anomaly Detection: Flag abnormal patterns in real-time sensor data.

3. Dashboard Readiness: Output-ready design to connect with a GUI or embedded interface.

# Presented  Solution for problem

The project is built around a simple, modular Python pipeline that simulates real-time monitoring of mechanical components using data like temperature, vibrations, and usage time.

1. Data Simulation:
Simulated sensor data is used to mimic real-time system behavior in the absence of physical hardware.

2. Preprocessing:
The raw logs are cleaned and structured to ensure consistency and readiness for analysis.

3. Fault Detection Logic:
Rule-based checks are applied to flag abnormal conditions like:

High temperature , Sudden vibration spikes , Extended usage beyond safe limits

4. ML Integration :
Machine learning models (e.g., Random Forest, Isolation Forest) are tested on the data to predict failures and detect anomalies.

5. Modular Design:
The code is kept flexible for future integration with GUI dashboards or real-time embedded systems.


# Tools & Technologies
Python (Pandas, NumPy, Scikit-learn, Matplotlib)

Simulated Sensor Data

Future-ready for GUI or Embedded Deployment


