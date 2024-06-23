import pandas as pd
import numpy as np


# import matplotlib.pyplot as plt


def distribute(center: float, w1: float, w2: float, width, arr):
    """
      This function distributes the weights over a range of airfoil linearly centered around a given value.

      Args:
            center: the most weighted value located at the center
            w1: the weight of the most weighted value
            w2: the weight of the least weighted values
            width: the width of the range of airfoil linearly centered around a given value, number of points of the array "arr"
            arr: array of all angle of attacks

      Returns:
            The array of weights of the airfoil linearly centered around a given value with the same shape as the given array "arr"
      """
    dy_dx = (w1 - w2) / width
    weights = np.ones(arr.shape) * w2
    center_i = np.searchsorted(arr, center)
    arr2 = arr[center_i - width: center_i + width]
    min_i = center_i - width
    for i in range(len(arr)):
        if abs(i - center_i) <= width:
            if i <= center_i:
                weights[i] = w2 + dy_dx * (i - min_i)
            elif i > center_i:
                weights[i] = w1 - dy_dx * (i - center_i)
    return weights


def weighted_mean(arr, weights):
    """Returns the weighted mean given an array and its weights"""
    return np.average(arr, weights=weights)


def max_val_dict(my_dict):
    """Returns the key and value of the maximum value in a dictionary"""
    max_item = max(my_dict.items(), key=lambda item: item[1])
    return max_item[0], max_item[1]


def sort_dict(my_dict):
    """Returns a dictionary sorted by value"""
    return dict(sorted(my_dict.items(), key=lambda item: item[1]))

def delete_zero_values(dict_in):
    """Returns a dictionary without zero values"""
    return {key: value for key, value in dict_in.items() if value != 0}

df = pd.read_pickle('data.pkl')
cl = df['cl'].to_numpy()
cd = df['cd'].to_numpy()
cm = df['cm'].to_numpy()
cl_cd = df['cl_cd'].to_numpy()
top_xtr = df['top_xtr'].to_numpy()
bot_xtr = df['bot_xtr'].to_numpy()
mach_crit = df['mach_crit'].to_numpy()
names = df['name'].to_numpy()
thickness = df['thickness'].to_numpy()
camber = df['camber'].to_numpy()

Re = np.array([1e4, 5e4, 1e5, 3e5, 5e5, 7e5, 1e6])
alpha = np.linspace(-10, 20, 301)

# Range and weights of Re and alpha
Re_range = np.array([1e4, 5e4, 3e5])
Re_weights2 = np.array([1, 1, 1])
Re_weights = np.ones(len(Re))
min_aoa = 0
min_aoa_i = np.searchsorted(alpha, min_aoa)
max_aoa = 15
max_aoa_i = np.searchsorted(alpha, max_aoa)
aoa_range = alpha[min_aoa_i:max_aoa_i + 1]
aoa_weights2 = distribute(2, 3, 2, 10, aoa_range)
aoa_weights = np.zeros(alpha.shape)
aoa_weights[min_aoa_i:max_aoa_i + 1] = aoa_weights2


def max_cl(Re_range, re_weights, aoa_weights):
    cl_means = {}
    for i in range(len(names)):
        cl_res_aoas = np.array(cl[i])
        cl_res = np.zeros(len(Re_range))
        j2 = 0
        for j in range(len(cl_res_aoas)):
            if Re[j] in Re_range:
                cl_res[j2] = weighted_mean(cl_res_aoas[j], aoa_weights)
                j2 += 1
            else:
                pass
        cl_means[names[i]] = weighted_mean(cl_res, re_weights)
    return sort_dict(delete_zero_values(cl_means))


def check_cl(cl_values, cl_array):
    tol = 0.03
    if (cl_values[0] - cl_array[0]) < tol or (cl_values[-1] - cl_array[-1]) > tol:
        return False
    else:
        return True

def check_cl_aoa(cl_values, cl_array, aoa_range):
    min_aoa = np.searchsorted(alpha, aoa_range[0])
    max_aoa = np.searchsorted(alpha, aoa_range[-1])
    if check_cl(cl_values, cl_array):
        i = np.searchsorted(cl_array, cl_values)
        if i[0] < min_aoa or i[-1] > max_aoa:
            return False
        else:
            return True
    else:
        return False



def interp_cl_cd(Cl_values, Cl_array, Cd_array):
    Cd_values = np.zeros(Cl_values.shape)
    for i, Cl_value in np.ndenumerate(Cl_values):
        Cl_index = np.searchsorted(Cl_array, Cl_value)
        if Cl_index == 0:
            Cd_values[i] = Cd_array[0]
        elif Cl_index == len(Cl_array):
            Cd_values[i] = Cd_array[-1]
        else:
            Cl_lower = Cl_array[Cl_index - 1]
            Cl_upper = Cl_array[Cl_index]
            Cd_lower = Cd_array[Cl_index - 1]
            Cd_upper = Cd_array[Cl_index]

            Cd_interp = Cd_lower + (Cl_value - Cl_lower) * (Cd_upper - Cd_lower) / (Cl_upper - Cl_lower)
            Cd_values[i] = Cd_interp
    return Cd_values


def min_cd_cl(Re_range, Re_weights, cl_range, cl_weights):
    cd_means = {}  # stores the weighted means of CD of all airfoils
    for i in range(len(names)):
        cd_res = np.zeros(Re_range.shape)  # stores weighted mean of cd calculated in the CL_range in each Re in Re_range
        j2 = 0
        for j in range(len(Re)):  # loops through all Re elements
            if Re[j] in Re_range:
                if check_cl(cl_range, cl[i][j]):
                    cd_cls = interp_cl_cd(cl_range, cl[i][j], cd[i][j])  # stores the values of cd corresponding to each Cl in Cl_range
                    cd_res[j2] = weighted_mean(cd_cls, cl_weights)
                    j2 += 1
                else:
                    pass
            else:
                pass

        cd_means[names[i]] = weighted_mean(cd_res, Re_weights)
    return sort_dict(delete_zero_values(cd_means))

def min_cd(Re_range, re_weights, cl_range, cl_weights, aoa_range):
    cd_means = {}  # stores the weighted means of CD of all airfoils
    for i in range(len(names)):
        cd_res = np.zeros(Re_range.shape)  # stores weighted mean of cd calculated in the CL_range in each Re in Re_range
        j2 = 0
        for j in range(len(Re)):  # loops through all Re elements
            if Re[j] in Re_range:
                if check_cl_aoa(cl_range, cl[i][j], aoa_range):
                    cd_cls = interp_cl_cd(cl_range, cl[i][j], cd[i][j])  # stores the values of cd corresponding to each Cl in Cl_range
                    cd_res[j2] = weighted_mean(cd_cls, cl_weights)
                    j2 += 1
                else:
                    pass
            else:
                pass
        cd_means[names[i]] = weighted_mean(cd_res, re_weights)
    return sort_dict(delete_zero_values(cd_means))

def max_cl_cd(Re_range, re_weights,aoa_weights):
    cl_cd_means = {}
    for i in range(len(names)):
        cl_res_aoas = np.array(cl[i])
        cl_res = np.zeros(len(Re_range))
        j2 = 0
        for j in range(len(cl_res_aoas)):
            if Re[j] in Re_range:
                cl_res[j2] = weighted_mean(cl_res_aoas[j], aoa_weights)
                j2 += 1
            else:
                pass
        cl_cd_means[names[i]] = weighted_mean(cl_res, re_weights)
    return sort_dict(delete_zero_values(cl_cd_means))

# constraints
def constraint_thickness(dict, thick):
    dictt = dict
    for key in enumerate(dictt.keys()):
        if thickness[key[0]] in thick:
            pass
        else:
            del dictt[key[1]]
    return dictt

def constraint_camber(dict, camb):
    dictt = dict
    for key in enumerate(dictt.keys()):
        if camber[key[0]] in camb:
            pass
        else:
            del dictt[key[1]]
    return dictt

