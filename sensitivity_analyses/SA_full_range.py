#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 19 12:00:28 2024

@author: williamtaylor
"""

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

problem = {
    'num_vars': 6,
    'names': ['theta', 'rho_b','dispersivity','lamb','alpha','kd'],
    'bounds': [[0.25, 0.7], # theta
               [0.29, 1.74], # rho_b
               [np.log10(2e-3), np.log10(1800)], # dispersivity
               [np.log10(5e-4), np.log10(500)], # lamb
               [np.log10(0.01), np.log10(24)], # alpha
               [np.log10(0.01), np.log10(100)]] # kd
}

param_values = saltelli.sample(problem, 2**13)
times = np.linspace(0,50,1000)

# static parameters, consistent between both models
Co = 1
L = 2
x=2
ts=0.25
v=1

param_values[:,2] = 10**param_values[:,2]
param_values[:,3] = 10**param_values[:,3]
param_values[:,4] = 10**param_values[:,4]
param_values[:,5] = 10**param_values[:,5]

# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# evaluate model 106
Y_early = np.zeros([param_values.shape[0]])
Y_peak = np.zeros([param_values.shape[0]])
Y_late = np.zeros([param_values.shape[0]])

# initiaze a list for the btcs
btc_data = []

# evaluate the model at the sampled parameter values, across the same dimensionless time and fixed L
for i,X in tqdm(enumerate(param_values), desc='Running Analysis Model 106'):
    concentrations, adaptive_times = model.concentration_106_new_adaptive_extended(times,X[0],X[1],X[2],X[3],X[4],X[5],Co=Co, v=v, ts=ts, L=L, x=x)

    Y_early[i], Y_peak[i], Y_late[i] = model.calculate_metrics(adaptive_times, concentrations)

    btc_data.append({
        'index':i,
        'params':X.tolist(),
        'times':adaptive_times,
        'concentrations':concentrations.tolist()
        })

metrics_df = pd.DataFrame({
    'Early': Y_early,
    'Peak': Y_peak,
    'Late': Y_late
})

metrics_df.to_csv(os.path.join(output_dir, 'metrics_106.csv'), index=True)

# save BTCs to json
with open(os.path.join(output_dir, 'btc_data_model106.json'), 'w') as f:
    json.dump(btc_data, f)

# apply Sobol method to each set of results    
Si_early = sobol.analyze(problem, Y_early, print_to_console=False)
Si_peak = sobol.analyze(problem, Y_peak, print_to_console=False)
Si_late = sobol.analyze(problem, Y_late, print_to_console=False)

total_Si_early, first_Si_early, second_Si_early = Si_early.to_df()
total_Si_peak, first_Si_peak, second_Si_peak = Si_peak.to_df()
total_Si_late, first_Si_late, second_Si_late = Si_late.to_df()

# save results
total_Si_early.to_csv(os.path.join(output_dir, 'total_Si_early_106.csv'), index=True)
first_Si_early.to_csv(os.path.join(output_dir, 'first_Si_early_106.csv'), index=True)
second_Si_early.to_csv(os.path.join(output_dir, 'second_Si_early_106.csv'), index=True)
total_Si_peak.to_csv(os.path.join(output_dir, 'total_Si_peak_106.csv'), index=True)
first_Si_peak.to_csv(os.path.join(output_dir, 'first_Si_peak_106.csv'), index=True)
second_Si_peak.to_csv(os.path.join(output_dir, 'second_Si_peak_106.csv'), index=True)
total_Si_late.to_csv(os.path.join(output_dir, 'total_Si_late_106.csv'), index=True)
first_Si_late.to_csv(os.path.join(output_dir, 'first_Si_late_106.csv'), index=True)
second_Si_late.to_csv(os.path.join(output_dir, 'second_Si_late_106.csv'), index=True)


