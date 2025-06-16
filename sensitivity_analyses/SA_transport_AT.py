import sys
import os
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(parent_dir)
import model
import numpy as np
import pandas as pd
from mpmath import invertlaplace, mp
mp.dps = 12
from SALib.sample import saltelli
from SALib.analyze import sobol
import json
from tqdm import tqdm

base_dir = os.path.dirname(os.path.abspath(__file__))
output_dir = os.path.normpath(os.path.join(base_dir, '..', 'results'))
os.makedirs(output_dir, exist_ok=True)

# advective controlled transport 
problem = {
    'num_vars': 6,
    'names': ['theta', 'rho_b','dispersivity','lamb','alpha','kd'],
    'bounds': [[0.25, 0.7], # theta
               [0.29, 1.74], # rho_b
               [np.log10(2e-3), np.log10(10e-1)], # dispersivity
               [np.log10(5e-4), np.log10(3e-1)], # lamb
               [np.log10(0.01), np.log10(24)], # alpha
               [np.log10(0.01), np.log10(100)]] # kd
}
times = np.linspace(0,100,1000)
L = 2
x = 2
ts = 0.25
v = 1
Co = 1

param_values = saltelli.sample(problem, 2**13)

# convert log sampled parameters back to real space
param_values[:,2] = 10**param_values[:,2]
param_values[:,3] = 10**param_values[:,3]
param_values[:,4] = 10**param_values[:,4]
param_values[:,5] = 10**param_values[:,5]

params_df = pd.DataFrame(data=param_values,
                         columns=['theta', 'rho_b','dispersivity','lamb','alpha','kd'])

Y_early_at = np.zeros(param_values.shape[0])
Y_peak_at = np.zeros(param_values.shape[0])
Y_late_at = np.zeros(param_values.shape[0])

btc_data = []

for i, X in tqdm(enumerate(param_values), desc='Running Analysis'):
    concentrations, adaptive_times = model.concentration_106_new_adaptive_extended(times,X[0],X[1],X[2],X[3],X[4],X[5], Co=Co, v=v, ts=ts, L=L, x=x)

    Y_early_at[i], Y_peak_at[i], Y_late_at[i] = model.calculate_metrics(adaptive_times, concentrations)

    btc_data.append({
        "index": i,
        "params": X.tolist(),
        "times": adaptive_times,
        "concentrations": concentrations.tolist()
        })

metrics_df = pd.DataFrame({
    'Early': Y_early_at,
    'Peak': Y_peak_at,
    'Late': Y_late_at
})

# Save metrics DataFrame
metrics_df.to_csv(os.path.join(output_dir, 'metrics_at.csv'), index=True)

# Save BTC data as JSON
with open(os.path.join(output_dir, 'btc_data_at.json'), 'w') as f:
    json.dump(btc_data, f)

# Sobol analysis and saving results
Si_early_at = sobol.analyze(problem, Y_early_at, print_to_console=False)
Si_peak_at = sobol.analyze(problem, Y_peak_at, print_to_console=False)
Si_late_at = sobol.analyze(problem, Y_late_at, print_to_console=False)

total_Si_early_at, first_Si_early_at, second_Si_early_at = Si_early_at.to_df()
total_Si_peak_at, first_Si_peak_at, second_Si_peak_at = Si_peak_at.to_df()
total_Si_late_at, first_Si_late_at, second_Si_late_at = Si_late_at.to_df()

# Save all Sobol indices results
total_Si_early_at.to_csv(os.path.join(output_dir, 'total_Si_early_at.csv'), index=True)
first_Si_early_at.to_csv(os.path.join(output_dir, 'first_Si_early_at.csv'), index=True)
second_Si_early_at.to_csv(os.path.join(output_dir, 'second_Si_early_at.csv'), index=True)
total_Si_peak_at.to_csv(os.path.join(output_dir, 'total_Si_peak_at.csv'), index=True)
first_Si_peak_at.to_csv(os.path.join(output_dir, 'first_Si_peak_at.csv'), index=True)
second_Si_peak_at.to_csv(os.path.join(output_dir, 'second_Si_peak_at.csv'), index=True)
total_Si_late_at.to_csv(os.path.join(output_dir, 'total_Si_late_at.csv'), index=True)
first_Si_late_at.to_csv(os.path.join(output_dir, 'first_Si_late_at.csv'), index=True)
second_Si_late_at.to_csv(os.path.join(output_dir, 'second_Si_late_at.csv'), index=True)