import sys
import os
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(parent_dir)
import numpy as np
import pandas as pd
from scipy import special
import model
from SALib.sample import saltelli
from SALib.analyze import sobol
import json
from tqdm import tqdm
import os

base_dir = os.path.dirname(os.path.abspath(__file__))
output_dir = os.path.normpath(os.path.join(base_dir, '..', 'results'))
os.makedirs(output_dir, exist_ok=True)

# advective controlled reaction
problem = {
    'num_vars': 6,
    'names': ['theta', 'rho_b','dispersivity','lamb','alpha','kd'],
    'bounds': [[0.25, 0.7], # theta
               [0.29, 1.74], # rho_b
               [np.log10(2e-3), np.log10(10e-1)], # dispersivity
               [np.log10(8e-1), np.log10(500)], # lambda - first order decay rate constant
               [np.log10(0.01), np.log10(24)], # alpha - first order desorption rate constant
               [np.log10(0.01), np.log10(100)]] # kd - sorption distribution coefficient
}

# perform sampling
param_values = saltelli.sample(problem, 2**13)

# time discretization
times = np.linspace(0,100,1000)
L = 2 # length
x = 2 # reference point
ts = 0.25 # pulse duration (days)
v = 1 # pore velocity (m/day)
Co = 1

# convert log sampled parameters back to real space
param_values[:,2] = 10**param_values[:,2]
param_values[:,3] = 10**param_values[:,3]
param_values[:,4] = 10**param_values[:,4]
param_values[:,5] = 10**param_values[:,5]

params_df = pd.DataFrame(data=param_values,
                         columns=['theta', 'rho_b','dispersivity','lamb','alpha','kd'])

# make empty arrays for storing metrics
Y_early_ar = np.zeros(param_values.shape[0])
Y_peak_ar = np.zeros(param_values.shape[0])
Y_late_ar = np.zeros(param_values.shape[0])

# make a list for appending/storing breakthrough curve concentrations
btc_data = []

for i, X in tqdm(enumerate(param_values), desc='Running Analysis'):
    concentrations, adaptive_times = model.concentration_106_new_adaptive_extended(times,X[0],X[1],X[2],X[3],X[4],X[5], Co=Co, v=v, ts=ts, L=L, x=x)
    
    Y_early_ar[i], Y_peak_ar[i], Y_late_ar[i] = model.calculate_metrics(adaptive_times, concentrations)

    #print(f'Advection reaction iteration: {i}')

    btc_data.append({
        "index": i,
        "params": X.tolist(),
        "times": adaptive_times,
        "concentrations": concentrations.tolist()
        })

metrics_df = pd.DataFrame({
    'Early': Y_early_ar,
    'Peak': Y_peak_ar,
    'Late': Y_late_ar
})

# Save metrics DataFrame
metrics_df.to_csv(os.path.join(output_dir, 'metrics_ar.csv'), index=True)

# Save BTC data as JSON
with open(os.path.join(output_dir, 'btc_data_ar.json'), 'w') as f:
    json.dump(btc_data, f)

# Sobol analysis and saving results
Si_early_ar = sobol.analyze(problem, Y_early_ar, print_to_console=False)
Si_peak_ar = sobol.analyze(problem, Y_peak_ar, print_to_console=False)
Si_late_ar = sobol.analyze(problem, Y_late_ar, print_to_console=False)

total_Si_early_ar, first_Si_early_ar, second_Si_early_ar = Si_early_ar.to_df()
total_Si_peak_ar, first_Si_peak_ar, second_Si_peak_ar = Si_peak_ar.to_df()
total_Si_late_ar, first_Si_late_ar, second_Si_late_ar = Si_late_ar.to_df()

# Save all Sobol indices results
total_Si_early_ar.to_csv(os.path.join(output_dir, 'total_Si_early_ar.csv'), index=True)
first_Si_early_ar.to_csv(os.path.join(output_dir, 'first_Si_early_ar.csv'), index=True)
second_Si_early_ar.to_csv(os.path.join(output_dir, 'second_Si_early_ar.csv'), index=True)
total_Si_peak_ar.to_csv(os.path.join(output_dir, 'total_Si_peak_ar.csv'), index=True)
first_Si_peak_ar.to_csv(os.path.join(output_dir, 'first_Si_peak_ar.csv'), index=True)
second_Si_peak_ar.to_csv(os.path.join(output_dir, 'second_Si_peak_ar.csv'), index=True)
total_Si_late_ar.to_csv(os.path.join(output_dir, 'total_Si_late_ar.csv'), index=True)
first_Si_late_ar.to_csv(os.path.join(output_dir, 'first_Si_late_ar.csv'), index=True)
second_Si_late_ar.to_csv(os.path.join(output_dir, 'second_Si_late_ar.csv'), index=True)