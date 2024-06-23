import numpy
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pyxfoil as pxf
from pyxfoil import set_xfoilexe, set_workdir
from pathlib import Path
import json

set_workdir("results")
set_xfoilexe("E:/Downloads/Airfoil-Optimization-master/Airfoil-Optimization-master/xfoil/xfoil.exe")

def average_value(x_values, y_values):
  """
  Calculates the average value as area under curve (AUC) divided by domain length.

  Args:
      x_values: A list of x values representing the independent variable.
      y_values: A list of y values representing the dependent variable.

  Returns:
      The average value (AUC / domain length).
  """
  dx = (x_values[-1] - x_values[0])/len(x_values)
  area = dx*0.5*(y_values[0] + y_values[-1] + np.sum(y_values[1:-1])*2)
  average_value = area / (x_values[-1] - x_values[0])
  return average_value

filenames = []
cl = []
cd = []
cm = []
clcd=[]
res = [1e5, 3e5, 5e5]
for file in Path('airfoilsdb').glob("*"):
    try:
        airfoil = pxf.Xfoil('airfoil')
        path='airfoilsdb/'+file.name
        cll = []
        cdd = []
        cmm =[]
        clocd = []
        airfoil.points_from_dat(path)
        airfoil.set_ppar(180)
        for re in res:
            polar = airfoil.run_polar(0, 10, 2, mach=0.1, re=re)
            cll.append(polar.cl)
            cdd.append(polar.cd)
            cmm.append(polar.cm)
            clocd.append(polar.clocd)
        cl.append(cll)
        cd.append(cdd)
        cm.append(cmm)
        clcd.append(clocd)
        filenames.append(file.name)

    except:
        print('errorr:'+file.name)

print(cl)
print(cd)
print(filenames)

dict = {'filenames': filenames, 'cl': cl, 'cd': cd, 'cm': cm, 'clcd': clcd}
print(dict)

df = pd.DataFrame(dict)
df['cl'] = df['cl'].apply(json.dumps)
df['cd'] = df['cd'].apply(json.dumps)
df['cm'] = df['cm'].apply(json.dumps)
df['clcd'] = df['clcd'].apply(json.dumps)
df.to_csv('data.csv')