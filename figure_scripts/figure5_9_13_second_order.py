#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar  1 11:42:03 2025

@author: williamtaylor
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json
import itertools

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

#-------------------------------------------------------
#-------------------------------------------------------
#%% full range, early arrival (kd, ai, a)
plt.style.use('default')

# Load data
btc_data = read_result_json('btc_data_model106.json')

# Extract parameters into NumPy arrays
theta = np.array([entry['params'][0] for entry in btc_data])
bulk = np.array([entry['params'][1] for entry in btc_data])
dispersivity = np.array([entry['params'][2] for entry in btc_data])
decay = np.array([entry['params'][3] for entry in btc_data])
alpha = np.array([entry['params'][4] for entry in btc_data])
kd = np.array([entry['params'][5] for entry in btc_data])

# Store parameters in a dictionary for easy looping
param_dict = {
    'Theta': theta,
    'Bulk': bulk,
    'Dispersivity': dispersivity,
    'Decay': decay,
    'Alpha': alpha,
    'Kd': kd
}
time = 'early'
if time == 'early':
    time_idx = 0
    label = 'Early arrival (log)'
if time == 'peak':
    time_idx = 1
    label = 'Time to peak concentration (log)'
if time == 'late':
    time_idx = 2
    label = 'Late time tailing (log)'

metrics = read_result_csv('metrics_106.csv')
output_metric = metrics[:,time_idx] 

fig, axes = plt.subplots(1,3,figsize=(12,4))

axes = axes.flatten()

sc= axes[0].scatter(dispersivity, kd, c=np.log10(output_metric), cmap='viridis', s=3, vmin=-1, vmax=2.8)
axes[0].set_xlabel(r'$\alpha_i$', fontweight='bold', fontsize=16)
axes[0].set_ylabel(r'$K_d$', fontweight='bold', fontsize=16)
axes[0].set_xscale('log')
axes[0].set_yscale('log')
axes[0].set_title('(a)', fontweight='bold', fontsize=14, loc='left')

axes[1].scatter(alpha, kd, c=np.log10(output_metric), cmap='viridis', s=3, vmin=-1, vmax=2.8)
axes[1].set_xlabel(r'$\alpha$', fontweight='bold', fontsize=16)
axes[1].set_ylabel(r'$K_d$', fontweight='bold', fontsize=16)
axes[1].set_xscale('log')
axes[1].set_yscale('log')
axes[1].set_title('(b)', fontweight='bold', fontsize=14, loc='left')

axes[2].scatter(alpha, dispersivity, c=np.log10(output_metric), cmap='viridis', s=3, vmin=-1, vmax=2.8)
axes[2].set_xlabel(r'$\alpha$', fontweight='bold', fontsize=16)
axes[2].set_ylabel(r'$\alpha_i$', fontweight='bold', fontsize=16)
axes[2].set_xscale('log')
axes[2].set_yscale('log')
axes[2].set_title('(c)', fontweight='bold', fontsize=14, loc='left')

cbar_ax = fig.add_axes([0.92, 0.1, 0.02, 0.8])  # [left, bottom, width, height]

fig.colorbar(sc, cmap='viridis', cax=cbar_ax, label=label)
plt.tight_layout(rect=[0, 0, 0.9, 1])  # Leave space for colorbar

for ax in axes:
    ax.set_rasterized(True)

fig_path = os.path.join(fig_dir, f'{time}_full_range_second_order.pdf')
plt.savefig(fig_path, format='pdf', bbox_inches='tight')
plt.show()

#-------------------------------------------------------
#-------------------------------------------------------
#%% full range, time to peak concentration (kd, ai, pb)
time = 'peak'
if time == 'early':
    time_idx = 0
    label = 'Early arrival (log)'
if time == 'peak':
    time_idx = 1
    label = 'Time to peak concentration (log)'
if time == 'late':
    time_idx = 2
    label = 'Late time tailing (log)'

output_metric = metrics[:,time_idx] 

fig, axes = plt.subplots(1,3,figsize=(12,4))

axes = axes.flatten()

sc= axes[0].scatter(dispersivity, kd, c=np.log10(output_metric), cmap='viridis', s=3, vmin=-1, vmax=2.8)
axes[0].set_xlabel(r'$\alpha_i$', fontweight='bold', fontsize=16)
axes[0].set_ylabel(r'$K_d$', fontweight='bold', fontsize=16)
axes[0].set_xscale('log')
axes[0].set_yscale('log')
axes[0].set_title('(a)', fontweight='bold', fontsize=14, loc='left')

axes[1].scatter(bulk, kd, c=np.log10(output_metric), cmap='viridis', s=3, vmin=-1, vmax=2.8)
axes[1].set_xlabel(r'$\rho_b$', fontweight='bold', fontsize=16)
axes[1].set_ylabel(r'$K_d$', fontweight='bold', fontsize=16)
#axes[1].set_xscale('log')
axes[1].set_yscale('log')
axes[1].set_title('(b)', fontweight='bold', fontsize=14, loc='left')

axes[2].scatter(bulk, dispersivity, c=np.log10(output_metric), cmap='viridis', s=3, vmin=-1, vmax=2.8)
axes[2].set_xlabel(r'$\rho_b$', fontweight='bold', fontsize=16)
axes[2].set_ylabel(r'$\alpha_i$', fontweight='bold', fontsize=16)
#axes[2].set_xscale('log')
axes[2].set_yscale('log')
axes[2].set_title('(c)', fontweight='bold', fontsize=14, loc='left')

cbar_ax = fig.add_axes([0.92, 0.1, 0.02, 0.8])  # [left, bottom, width, height]

fig.colorbar(sc, cmap='viridis', cax=cbar_ax, label=label)
plt.tight_layout(rect=[0, 0, 0.9, 1])  # Leave space for colorbar

for ax in axes:
    ax.set_rasterized(True)

fig_path = os.path.join(fig_dir, f'{time}_full_range_second_order.pdf')
plt.savefig(fig_path, format='pdf', bbox_inches='tight')
plt.show()

#-------------------------------------------------------
#-------------------------------------------------------
# late time tailing
# kd, alpha_i, alpha, lambda, rho_b

time = 'late'
if time == 'early':
    time_idx = 0
if time == 'peak':
    time_idx = 1
if time == 'late':
    time_idx = 2

output_metric = metrics[:,time_idx] 

fig, axes = plt.subplots(2,5,figsize=(15,6))

axes = axes.flatten()

sc= axes[0].scatter(dispersivity, kd, c=np.log10(output_metric), cmap='viridis', s=3, vmin=-1, vmax=2.8)
axes[0].set_xlabel(r'$\alpha_i$', fontweight='bold', fontsize=16)
axes[0].set_ylabel(r'$K_d$', fontweight='bold', fontsize=16)
axes[0].set_xscale('log')
axes[0].set_yscale('log')
axes[0].set_title('(a)', fontweight='bold', fontsize=14, loc='left')

axes[1].scatter(alpha, kd, c=np.log10(output_metric), cmap='viridis', s=3, vmin=-1, vmax=2.8)
axes[1].set_xlabel(r'$\alpha$', fontweight='bold', fontsize=16)
axes[1].set_ylabel(r'$K_d$', fontweight='bold', fontsize=16)
axes[1].set_xscale('log')
axes[1].set_yscale('log')
axes[1].set_title('(b)', fontweight='bold', fontsize=14, loc='left')

axes[2].scatter(decay, kd, c=np.log10(output_metric), cmap='viridis', s=3, vmin=-1, vmax=2.8)
axes[2].set_xlabel(r'$\lambda$', fontweight='bold', fontsize=16)
axes[2].set_ylabel(r'$K_d$', fontweight='bold', fontsize=16)
axes[2].set_xscale('log')
axes[2].set_yscale('log')
axes[2].set_title('(c)', fontweight='bold', fontsize=14, loc='left')

axes[3].scatter(bulk, kd, c=np.log10(output_metric), cmap='viridis', s=3, vmin=-1, vmax=2.8)
axes[3].set_xlabel(r'$\rho_b$', fontweight='bold', fontsize=16)
axes[3].set_ylabel(r'$K_d$', fontweight='bold', fontsize=16)
#axes[3].set_xscale('log')
axes[3].set_yscale('log')
axes[3].set_title('(d)', fontweight='bold', fontsize=14, loc='left')

axes[4].scatter(alpha, dispersivity, c=np.log10(output_metric), cmap='viridis', s=3, vmin=-1, vmax=2.8)
axes[4].set_xlabel(r'$\alpha$', fontweight='bold', fontsize=16)
axes[4].set_ylabel(r'$\alpha_i$', fontweight='bold', fontsize=16)
axes[4].set_xscale('log')
axes[4].set_yscale('log')
axes[4].set_title('(e)', fontweight='bold', fontsize=14, loc='left')

axes[5].scatter(decay, dispersivity, c=np.log10(output_metric), cmap='viridis', s=3, vmin=-1, vmax=2.8)
axes[5].set_xlabel(r'$\lambda$', fontweight='bold', fontsize=16)
axes[5].set_ylabel(r'$\alpha_i$', fontweight='bold', fontsize=16)
axes[5].set_xscale('log')
axes[5].set_yscale('log')
axes[5].set_title('(f)', fontweight='bold', fontsize=14, loc='left')

axes[6].scatter(bulk, dispersivity, c=np.log10(output_metric), cmap='viridis', s=3, vmin=-1, vmax=2.8)
axes[6].set_xlabel(r'$\rho_b$', fontweight='bold', fontsize=16)
axes[6].set_ylabel(r'$\alpha_i$', fontweight='bold', fontsize=16)
#axes[6].set_xscale('log')
axes[6].set_yscale('log')
axes[6].set_title('(g)', fontweight='bold', fontsize=14, loc='left')

axes[7].scatter(decay, alpha, c=np.log10(output_metric), cmap='viridis', s=3, vmin=-1, vmax=2.8)
axes[7].set_xlabel(r'$\lambda$', fontweight='bold', fontsize=16)
axes[7].set_ylabel(r'$\alpha$', fontweight='bold', fontsize=16)
axes[7].set_xscale('log')
axes[7].set_yscale('log')
axes[7].set_title('(h)', fontweight='bold', fontsize=14, loc='left')

axes[8].scatter(bulk, alpha, c=np.log10(output_metric), cmap='viridis', s=3, vmin=-1, vmax=2.8)
axes[8].set_xlabel(r'$\rho_b$', fontweight='bold', fontsize=16)
axes[8].set_ylabel(r'$\alpha$', fontweight='bold', fontsize=16)
#axes[8].set_xscale('log')
axes[8].set_yscale('log')
axes[8].set_title('(i)', fontweight='bold', fontsize=14, loc='left')

axes[9].scatter(bulk, decay, c=np.log10(output_metric), cmap='viridis', s=3, vmin=-1, vmax=2.8)
axes[9].set_xlabel(r'$\rho_b$', fontweight='bold', fontsize=16)
axes[9].set_ylabel(r'$\lambda$', fontweight='bold', fontsize=16)
#axes[9].set_xscale('log')
axes[9].set_yscale('log')
axes[9].set_title('(j)', fontweight='bold', fontsize=14, loc='left')

cbar_ax = fig.add_axes([0.92, 0.1, 0.02, 0.8])  # [left, bottom, width, height]
fig.colorbar(sc, cmap='viridis', cax=cbar_ax, label="Late time tailing (log)")
plt.tight_layout(rect=[0, 0, 0.9, 1])  # Leave space for colorbar

for ax in axes:
    ax.set_rasterized(True)

fig_path = os.path.join(fig_dir, f'{time}_full_range_second_order.pdf')
plt.savefig(fig_path, format='pdf', bbox_inches='tight')
plt.show()


