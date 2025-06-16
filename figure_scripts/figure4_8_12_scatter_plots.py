#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  1 15:59:00 2025

@author: williamtaylor
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
from statsmodels.nonparametric.smoothers_lowess import lowess
plt.style.use('default')

base_dir = os.path.dirname(os.path.abspath(__file__))
results_dir = os.path.normpath(os.path.join(base_dir, '..', 'results'))
fig_dir = os.path.normpath(os.path.join(base_dir, '..', 'figures'))

# helper functions
def read_result_csv(filename):
    return pd.read_csv(os.path.join(results_dir, filename), index_col=0).to_numpy()

def read_result_json(filename):
    with open(os.path.join(results_dir, filename), 'r') as f:
        data = json.load(f)
    return data

time = 'early'

if time == 'early':
    time_idx = 0
if time == 'peak':
    time_idx = 1
if time == 'late':
    time_idx = 2

metrics = read_result_csv('metrics_106.csv')

btc_data = read_result_json('btc_data_model106.json')

# Extract parameters into NumPy arrays
theta = np.array([entry['params'][0] for entry in btc_data])
bulk = np.array([entry['params'][1] for entry in btc_data])
dispersivity = np.array([entry['params'][2] for entry in btc_data])
decay = np.array([entry['params'][3] for entry in btc_data])
alpha = np.array([entry['params'][4] for entry in btc_data])
kd = np.array([entry['params'][5] for entry in btc_data])

fig, axes = plt.subplots(1,3, figsize=(12,4), sharey=True)
axes = axes.flatten()
greek_labels = [r'$\theta$', r'$\rho_b$', r'$\alpha$', r'$\lambda$', r'$\alpha_i$', r'$K_d$']

axes[0].scatter(kd, metrics[:,time_idx])
axes[0].set_xscale('log')
axes[0].set_ylabel('Early arrival \n(dimensionless time)', fontweight='bold', fontsize=14)
axes[0].set_xlabel(r'$K_d$', fontsize=16)
axes[0].set_title('(a)', fontweight='bold', fontsize=16, loc='left')

axes[1].scatter(dispersivity, metrics[:,time_idx])
axes[1].set_xscale('log')
axes[1].set_title('(b)', fontweight='bold', fontsize=16, loc='left')
axes[1].set_xlabel(r'$\alpha_i$', fontsize=16)

axes[2].scatter(alpha, metrics[:,time_idx])
axes[2].set_xscale('log')
axes[2].set_title('(c)', fontweight='bold', fontsize=16, loc='left')
axes[2].set_xlabel(r'$\alpha$', fontsize=16)

for ax in axes:
    ax.set_rasterized(True)

plt.tight_layout()
fig_path = os.path.join(fig_dir, f'{time}_full_range_scatter.pdf')
plt.savefig(fig_path, format='pdf', bbox_inches='tight')
plt.show()

#---------------------------------------------
#---------------------------------------------
#%% time to peak concentration
time = 'peak'

if time == 'early':
    time_idx = 0
if time == 'peak':
    time_idx = 1
if time == 'late':
    time_idx = 2


# Extract parameters into NumPy arrays
theta = np.array([entry['params'][0] for entry in btc_data])
bulk = np.array([entry['params'][1] for entry in btc_data])
dispersivity = np.array([entry['params'][2] for entry in btc_data])
decay = np.array([entry['params'][3] for entry in btc_data])
alpha = np.array([entry['params'][4] for entry in btc_data])
kd = np.array([entry['params'][5] for entry in btc_data])

fig, axes = plt.subplots(1,3, figsize=(12,4), sharey=True)
axes = axes.flatten()
greek_labels = [r'$\theta$', r'$\rho_b$', r'$\alpha$', r'$\lambda$', r'$\alpha_i$', r'$K_d$']

axes[0].scatter(kd, metrics[:,time_idx])
axes[0].set_xscale('log')
axes[0].set_ylabel('Time to peak concentration \n(dimensionless time)', fontweight='bold', fontsize=14)
axes[0].set_xlabel(r'$K_d$', fontsize=16)
axes[0].set_title('(a)', fontweight='bold', fontsize=16, loc='left')

axes[1].scatter(dispersivity, metrics[:,time_idx])
axes[1].set_xscale('log')
axes[1].set_title('(b)', fontweight='bold', fontsize=16, loc='left')
axes[1].set_xlabel(r'$\alpha_i$', fontsize=16)

axes[2].scatter(bulk, metrics[:,time_idx])
#axes[2].set_xscale('log')
axes[2].set_title('(c)', fontweight='bold', fontsize=16, loc='left')
axes[2].set_xlabel(r'$\rho_b$', fontsize=16)

for ax in axes:
    ax.set_rasterized(True)

plt.tight_layout()
fig_path = os.path.join(fig_dir, f'{time}_full_range_scatter.pdf')
plt.savefig(fig_path, format='pdf', bbox_inches='tight')
plt.show()

#---------------------------------------------
#---------------------------------------------
#%% late time tailing
time = 'late'

if time == 'early':
    time_idx = 0
if time == 'peak':
    time_idx = 1
if time == 'late':
    time_idx = 2

# Extract parameters into NumPy arrays
theta = np.array([entry['params'][0] for entry in btc_data])
bulk = np.array([entry['params'][1] for entry in btc_data])
dispersivity = np.array([entry['params'][2] for entry in btc_data])
decay = np.array([entry['params'][3] for entry in btc_data])
alpha = np.array([entry['params'][4] for entry in btc_data])
kd = np.array([entry['params'][5] for entry in btc_data])

fig, axes = plt.subplots(1,5, figsize=(12,4), sharey=True)
axes = axes.flatten()
greek_labels = [r'$\theta$', r'$\rho_b$', r'$\alpha$', r'$\lambda$', r'$\alpha_i$', r'$K_d$']

axes[0].scatter(kd, metrics[:,time_idx])
axes[0].set_xscale('log')
axes[0].set_ylabel('Late time tailing \n(dimensionless time)', fontweight='bold', fontsize=14)
axes[0].set_xlabel(r'$K_d$', fontsize=16)
axes[0].set_title('(a)', fontweight='bold', fontsize=16, loc='left')

axes[1].scatter(alpha, metrics[:,time_idx])
axes[1].set_xscale('log')
axes[1].set_title('(b)', fontweight='bold', fontsize=16, loc='left')
axes[1].set_xlabel(r'$\alpha$', fontsize=16)

axes[2].scatter(dispersivity, metrics[:,time_idx])
axes[2].set_xscale('log')
axes[2].set_title('(c)', fontweight='bold', fontsize=16, loc='left')
axes[2].set_xlabel(r'$\alpha_i$', fontsize=16)

axes[3].scatter(decay, metrics[:,time_idx])
axes[3].set_xscale('log')
axes[3].set_title('(d)', fontweight='bold', fontsize=16, loc='left')
axes[3].set_xlabel(r'$\lambda$', fontsize=16)

axes[4].scatter(bulk, metrics[:,time_idx])
#axes[4].set_xscale('log')
axes[4].set_title('(e)', fontweight='bold', fontsize=16, loc='left')
axes[4].set_xlabel(r'$\rho_b$', fontsize=16)

for ax in axes:
    ax.set_rasterized(True)

plt.tight_layout()
fig_path = os.path.join(fig_dir, f'{time}_full_range_scatter.pdf')
plt.savefig(fig_path, format='pdf', bbox_inches='tight')
plt.show()