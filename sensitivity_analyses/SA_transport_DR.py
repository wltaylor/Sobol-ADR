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

# dispersion controlled reaction 
problem = {
    'num_vars': 6,
    'names': ['theta', 'rho_b','dispersivity','lamb','alpha','kd'],
    'bounds': [[0.25, 0.7], # theta
               [0.29, 1.74], # rho_b
               [np.log10(4), np.log10(1800)], # dispersivity
               [np.log10(8e-1), np.log10(500)], # lamb
               [np.log10(0.01), np.log10(24)], # alpha
               [np.log10(0.01), np.log10(100)]] # kd
}
times = np.linspace(0,10,1000)
L = 2
x = 2
ts = 0.25
v = 1
Co = 1

param_values = saltelli.sample(problem, 2**13)

param_values[:,2] = 10**param_values[:,2]
param_values[:,3] = 10**param_values[:,3]
param_values[:,4] = 10**param_values[:,4]
param_values[:,5] = 10**param_values[:,5]

params_df = pd.DataFrame(data=param_values,
                         columns=['theta', 'rho_b','dispersivity','lamb','alpha','kd'])

Y_early_dr = np.zeros(param_values.shape[0])
Y_peak_dr = np.zeros(param_values.shape[0])
Y_late_dr = np.zeros(param_values.shape[0])

btc_data = []

for i, X in tqdm(enumerate(param_values), desc='Running Analysis'):
    concentrations, adaptive_times = model.concentration_106_new_adaptive_extended(times,X[0],X[1],X[2],X[3],X[4],X[5], Co=Co, v=v, ts=ts, L=L, x=x)
    
    Y_early_dr[i], Y_peak_dr[i], Y_late_dr[i] = model.calculate_metrics(adaptive_times, concentrations)

    #print(f'Diffusion reaction iteration: {i}')

    btc_data.append({
        "index": i,
        "params": X.tolist(),
        "times": adaptive_times,
        "concentrations": concentrations.tolist()
        })

metrics_df = pd.DataFrame({
    'Early': Y_early_dr,
    'Peak': Y_peak_dr,
    'Late': Y_late_dr
})

# Save metrics DataFrame
metrics_df.to_csv(os.path.join(output_dir, 'metrics_dr.csv'), index=True)

# Save BTC data as JSON
with open(os.path.join(output_dir, 'btc_data_dr.json'), 'w') as f:
    json.dump(btc_data, f)

# Sobol analysis and saving results
Si_early_dr = sobol.analyze(problem, Y_early_dr, print_to_console=False)
Si_peak_dr = sobol.analyze(problem, Y_peak_dr, print_to_console=False)
Si_late_dr = sobol.analyze(problem, Y_late_dr, print_to_console=False)

total_Si_early_dr, first_Si_early_dr, second_Si_early_dr = Si_early_dr.to_df()
total_Si_peak_dr, first_Si_peak_dr, second_Si_peak_dr = Si_peak_dr.to_df()
total_Si_late_dr, first_Si_late_dr, second_Si_late_dr = Si_late_dr.to_df()

# Save all Sobol indices results
total_Si_early_dr.to_csv(os.path.join(output_dir, 'total_Si_early_dr.csv'), index=True)
first_Si_early_dr.to_csv(os.path.join(output_dir, 'first_Si_early_dr.csv'), index=True)
second_Si_early_dr.to_csv(os.path.join(output_dir, 'second_Si_early_dr.csv'), index=True)
total_Si_peak_dr.to_csv(os.path.join(output_dir, 'total_Si_peak_dr.csv'), index=True)
first_Si_peak_dr.to_csv(os.path.join(output_dir, 'first_Si_peak_dr.csv'), index=True)
second_Si_peak_dr.to_csv(os.path.join(output_dir, 'second_Si_peak_dr.csv'), index=True)
total_Si_late_dr.to_csv(os.path.join(output_dir, 'total_Si_late_dr.csv'), index=True)
first_Si_late_dr.to_csv(os.path.join(output_dir, 'first_Si_late_dr.csv'), index=True)
second_Si_late_dr.to_csv(os.path.join(output_dir, 'second_Si_late_dr.csv'), index=True)