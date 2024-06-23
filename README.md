# Multi-Objective Airfoil Optimization for UAV Design
This project implements an optimization algorithm for selecting airfoils that meet multiple design objectives for Unmanned Aerial Vehicles (UAVs). It focuses on maximizing lift coefficient (CL), minimizing drag coefficient (Cd), and optimizing the Cd/CL ratio, considering user-defined constraints.

# Features
- Supports multi-objective optimization with three objectives:
    - Maximizing lift coefficient (CL)
    - Minimizing drag coefficient (Cd)
    - Optimizing Cd/CL ratio
- Allows user-defined weights for each objective to prioritize specific design requirements.
- Considers optional constraints on maximum airfoil thickness and camber.
- Accepts a range of Reynolds numbers (Re) and angles of attack (AoA) with corresponding weights.