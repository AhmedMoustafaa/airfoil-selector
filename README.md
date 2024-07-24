# Multi-Objective Airfoil Selector for UAV Design
This project implements an optimized search algorithm for selecting airfoils that meet multiple design objectives for Unmanned Aerial Vehicles (UAVs). It focuses on maximizing the lift coefficient (CL), minimizing the drag coefficient (Cd), and optimizing the Cd/CL ratio, considering user-defined constraints.

# Features
- Supports searching with three objectives:
    - Maximizing lift coefficient (CL)
    - Minimizing drag coefficient (Cd)
    - Maximizing Cd/CL ratio
- Allows user-defined weights for each objective to prioritize specific design requirements.
- Considers optional constraints on maximum airfoil thickness and camber.
- Accepts a range of Reynolds numbers (Re) and angles of attack (AoA) with corresponding weights.

# Inputs
Depending on the objective of the search, different inputs are needed 
## Reynolds Numbers
- Pick suitable Reynolds numbers from the values [1e4, 5e4, 1e5, 3e5, 5e5, 7e5, 1e6]
- Specify weights for the picked values
## Angle of attack
- Pick suitable subrange of AoAs from the range (-10 to 20) degrees
- Specify weights for the picked values
- You can use the function "distribute" to linearly distribute a weight over a range of angles of attacks around a specific point
## Cl
- Pick the desired Cl range
- Specify weights for the picked values

# Usage
- Clone the git repo and save it in the directory of your project
- unzip "arifoilsdb"
- Download the required libraries [pandas, aerosandbox, numpy]
- run **main_ne.py**
- import **.analysis** to your code

## available functions
- distribute: This function distributes the weights over a range of AOAs linearly centered around a given value.
- max_cl: returns a dictionary of airfoil names as keys and the mean Cl as values sorted in ascending order
- max_cl_cd: returns a dictionary of airfoil names as keys and the mean Cl/Cd as values sorted in ascending order
- min_cd: returns a dictionary of airfoil names as keys and the mean Cd as values sorted in ascending order
- min_cd_aoa: returns a dictionary of airfoil names as keys and the mean Cd as values sorted in ascending order
- contraint_thickness: returns a dictionary of airfoils whose max thickness is within a given range
- constraint_camber: returns a dictionary of airfoils whose max camber is within a given range
## Examples
```python
Re_range = [1e4, 5e4, 1e5, 3e5, 5e5]
Re_weights = [3,2,2,5,2]  #assuming you take off at 1e4 and cruise at 3e5

aoa_range = np.linspace(-10,20,301)
aoa_weights = distribute(center=2, w1=5, w2=1, arr=aoa_range) #assuming you cruise at aoa = 2 degrees

# get maximum cl
maximum_cl = max_cl(Re_range, Re_weights, aoa_range, aoa_weights)
best_airfoils = list(maximum_cl.items())[-5:]
maximum_cl_constrained = constraint_camber(constraint_thickness(maximum_cl, thick_min=0.1, thick_max=0.25),camb_min=0, camb_max=0.02)  #0 <= camber < 2% of the chord, 10% <= thickness<=25% of the chord

# get the minimum cd
cl_range = [0.5,0.6,0.7,0.8]
cl_weights =  [1,2,3,4]
aoa_range = [0, 5]  #this will search for airfoils that have the given cl values only at the given aoa_range
min_cd = min_cd_aoa(Re_range, Re_weights, cl_range, cl_weights, aoa_range)
```

# TO-DO List
- [ ] make a website using Flask library that provides an easier interface for users
- [ ] optimize the workflow of the program (requires some knowledge that I don't currently have)
