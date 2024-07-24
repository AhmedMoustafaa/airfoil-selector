import pandas as pd
import aerosandbox as asb
import aerosandbox.numpy as np
from pathlib import Path
import warnings
import re

filenames = []
with warnings.catch_warnings(record=True) as warning_list:
    warnings.simplefilter("always")  # Capture all warnings
    for file in Path('airfoilsdb').glob("*"):
        airfoil = asb.Airfoil(file.name)
        filenames.append(file.name)

if warning_list:
    for warning in warning_list:
        match = re.search(r"(?<=Airfoil )([^ ]+\.dat)", str(warning.message))
        if match:
            print(match.group(1))
            filenames.remove(match.group(1))


airfoils = np.array(filenames)

alpha = np.linspace(-10,20,301)
re = np.array([1e4,5e4,1e5,3e5,5e4,7e5,1e6])
Alpha, Re = np.meshgrid(alpha,re)


shape = (np.length(airfoils), 7, 301)
cl = np.zeros(shape)
cd = np.zeros(shape)
cm = np.zeros(shape)
cl_cd = np.zeros(shape)
top_xtr = np.zeros(shape)
bot_xtr = np.zeros(shape)
mach_crit = np.zeros(shape)
max_thickness = np.zeros(shape)
camber = np.zeros(shape)

for i in range(len(airfoils)):
    af = asb.Airfoil(airfoils[i])
    progress = i/len(airfoils)
    print(f"calculating for airfoil:{af}...progress={progress} %")
    aero_data = af.get_aero_from_neuralfoil(
        alpha=Alpha.flatten(),
        Re=Re.flatten(),
        mach=0.0
    )
    Aero = {
        key: value.reshape(Alpha.shape)
        for key, value in aero_data.items()
    }
    cl[i] = Aero['CL']
    cd[i] = Aero['CD']
    cm[i] = Aero['CM']
    cl_cd[i] = Aero['CL']/Aero['CD']
    top_xtr[i] = Aero['Top_Xtr']
    bot_xtr[i] = Aero['Bot_Xtr']
    mach_crit[i] = Aero['mach_crit']
    max_thickness[i] = af.max_thickness()
    camber[i] = af.max_camber()

df = pd.DataFrame({
    'name':airfoils.tolist(),
    'cl':cl.tolist(),
    'cd':cd.tolist(),
    'cl_cd':cl_cd.tolist(),
    'cm':cm.tolist(),
    'top_xtr':top_xtr.tolist(),
    'bot_xtr':bot_xtr.tolist(),
    'mach_crit':mach_crit.tolist(),
    'thickness':max_thickness.tolist(),
    'camber':camber.tolist()
})

df.to_pickle('data.pkl')