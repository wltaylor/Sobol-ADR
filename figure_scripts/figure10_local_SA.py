#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 18 08:59:56 2025

@author: williamtaylor
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
import os
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(parent_dir)
import model

base_dir = os.path.dirname(os.path.abspath(__file__))
results_dir = os.path.normpath(os.path.join(base_dir, '..', 'results'))
fig_dir = os.path.normpath(os.path.join(base_dir, '..', 'figures'))

times = np.linspace(0,20,5000)
L = 2
x = 2
ts = 0.25
v = 1
Co = 1
theta = 0.5
rho_b = 1.0
high_dispersivity = np.linspace(np.log10(4),np.log10(1800),10) # create sample range in log-space
low_dispersivity = np.linspace(np.log10(2e-3),np.log10(10e-1),10) # create sample range in log-space
lamb = 0.01
alpha = 0.01
kd = 0.01

fig, ax = plt.subplots(2,3,figsize=(12,8))
ax = ax.flatten()
# simulate BTCs for low-dispersion scenarios first
tailings = []
for i,d in enumerate(low_dispersivity):
    concentrations, adaptive_times = model.concentration_106_new_adaptive_extended(times,theta,rho_b,10**d,lamb,alpha,kd, Co=Co, v=v, ts=ts, L=L, x=x) # ensure to convert d back to real space
    ax[0].plot(adaptive_times, concentrations, c='blue')

    _, _, tailing = model.calculate_metrics(adaptive_times, concentrations)
    tailings.append(tailing)
    
ax[2].scatter(10**low_dispersivity, tailings, c='blue')
print(f'Low dispersion tailing variance: {np.array(tailings).var():.3f}')

ax[0].set_xlabel('Time \n(Dimensionless)', fontweight='bold')
ax[0].set_ylabel('Concentration (C/Co)', fontweight='bold')
ax[0].set_title('(a)', loc='left', fontweight='bold', fontsize=14)
ax[0].set_ylim(0,0.75)

# simulate BTCs for high-dispersion scenarios next
tailings = []
for i,d in enumerate(high_dispersivity):
    concentrations, adaptive_times = model.concentration_106_new_adaptive_extended(times,theta,rho_b,10**d,lamb,alpha,kd, Co=Co, v=v, ts=ts, L=L, x=x) # ensure to convert d back to real space
    ax[1].plot(adaptive_times, concentrations, c='green')

    _, _, tailing = model.calculate_metrics(adaptive_times, concentrations)
    tailings.append(tailing)
ax[2].scatter(10**high_dispersivity, tailings, c='green')
print(f'High dispersion tailing variance: {np.array(tailings).var():.3f}')

ax[1].set_xlabel('Time \n(Dimensionless)', fontweight='bold')
ax[1].set_ylabel('Concentration (C/Co)', fontweight='bold')
ax[1].set_title('(b)', loc='left', fontweight='bold', fontsize=14)
ax[1].set_ylim(0,0.75)

ax[2].set_xlabel(r'Dispersivity ($\alpha_i$)', fontweight='bold')
ax[2].set_ylabel('Late-time tailing \n(Dimensionless time)', fontweight='bold')
ax[2].set_xscale('log')
ax[2].set_title('(c)', loc='left', fontweight='bold', fontsize=14)

L = 2
x = 2
ts = 0.25
v = 1
Co = 1
theta = 0.5
rho_b = 1.0
d = 0.1
low_lamb = np.linspace(np.log10(5e-4),np.log10(3e-1),10) # create sample range in log-space
high_lamb = np.linspace(np.log10(8e-1),np.log10(500),10) # create sample range in log-space
alpha = 0.01
kd = 0.01

# simulate BTCs for high-dispersion scenarios first
tailings = []
for i,lamb in enumerate(low_lamb):
    concentrations, adaptive_times = model.concentration_106_new_adaptive_extended(times,theta,rho_b,d,10**lamb,alpha,kd, Co=Co, v=v, ts=ts, L=L, x=x) # ensure to convert d back to real space
    ax[3].plot(adaptive_times, concentrations, c='blue')

    _, _, tailing = model.calculate_metrics(adaptive_times, concentrations)
    tailings.append(tailing)
ax[5].scatter(10**low_lamb, tailings, c='blue')
print(f'High dispersion tailing variance: {np.array(tailings).var():.3f}')

ax[3].set_xlabel('Time \n(Dimensionless)', fontweight='bold')
ax[3].set_ylabel('Concentration (C/Co)', fontweight='bold')
ax[3].set_title('(d)', loc='left', fontweight='bold', fontsize=14)
ax[3].set_ylim(0,0.2)

# simulate BTCs for low-dispersion scenarios next
tailings = []
for i,lamb in enumerate(high_lamb):
    concentrations, adaptive_times = model.concentration_106_new_adaptive_extended(times,theta,rho_b,d,10**lamb,alpha,kd, Co=Co, v=v, ts=ts, L=L, x=x) # ensure to convert d back to real space
    ax[4].plot(adaptive_times, concentrations, c='green')

    _, _, tailing = model.calculate_metrics(adaptive_times, concentrations)
    tailings.append(tailing)
    
ax[5].scatter(10**high_lamb, tailings, c='green')
print(f'Low dispersion tailing variance: {np.array(tailings).var():.3f}')

ax[4].set_xlabel('Time \n(Dimensionless)', fontweight='bold')
ax[4].set_ylabel('Concentration (C/Co)', fontweight='bold')
ax[4].set_title('(e)', loc='left', fontweight='bold', fontsize=14)
ax[4].set_ylim(0,0.2)

ax[5].set_xlabel(r'Decay ($\lambda$)', fontweight='bold')
ax[5].set_ylabel('Late-time tailing \n(Dimensionless time)', fontweight='bold')
ax[5].set_xscale('log')
ax[5].set_title('(f)', loc='left', fontweight='bold', fontsize=14)

plt.tight_layout()
fig_path = os.path.join(fig_dir, f'local_SA.pdf')
plt.savefig(fig_path, format='pdf', bbox_inches='tight')
plt.show()

