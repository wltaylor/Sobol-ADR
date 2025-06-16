#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 15 12:36:28 2025

@author: williamtaylor
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from SALib.sample import saltelli
import seaborn as sns
import matplotlib.patches as mpatches
from pandas.plotting import table
plt.style.use('ggplot')

base_dir = os.path.dirname(os.path.abspath(__file__))
results_dir = os.path.normpath(os.path.join(base_dir, '..', 'results'))
fig_dir = os.path.normpath(os.path.join(base_dir, '..', 'figures'))

# adv reaction
problem = {
    'num_vars': 6,
    'names': ['theta', 'rho_b','dispersivity','lamb','alpha','kd'],
    'bounds': [[0.25, 0.7], # theta - porosity
               [0.29, 1.74], # rho_b - bulk density
               [np.log10(2e-3), np.log10(10e-1)], # dispersivity
               [np.log10(8e-1), np.log10(500)], # lamb - first order decay rate constant
               [0, 24], # alpha - first order desorption rate constant
               [0.01, 100]] # kd - sorption distribution coefficient
}

param_values = saltelli.sample(problem, 2**8)

adv_reaction = pd.DataFrame(data=param_values,
                         columns=['theta', 'rho_b','dispersivity','lamb','alpha','kd'])
# convert back to real space
adv_reaction['dispersivity'] = 10**adv_reaction['dispersivity']
adv_reaction['lamb'] = 10**adv_reaction['lamb']

adv_reaction['v'] = 1 # m/day
# calculate dispersion
adv_reaction['D'] = adv_reaction['v'] * adv_reaction['dispersivity']
# calculate Peclet and Dahmkoler
adv_reaction['Pe'] = (adv_reaction['v'] * 2) / adv_reaction['D']
adv_reaction['Da'] = (adv_reaction['lamb'] * 2) / adv_reaction['v']

# advective transport
problem = {
    'num_vars': 6,
    'names': ['theta', 'rho_b','dispersivity','lamb','alpha','kd'],
    'bounds': [[0.25, 0.7], # theta - porosity
               [0.29, 1.74], # rho_b - bulk density
               [np.log10(2e-3), np.log10(10e-1)], # dispersivity
               [np.log10(5e-4), np.log10(3e-1)], # lamb - first order decay rate constant
               [0, 24], # alpha - first order desorption rate constant
               [0.01, 100]] # kd - sorption distribution coefficient
}

param_values = saltelli.sample(problem, 2**8)

adv_trans = pd.DataFrame(data=param_values,
                         columns=['theta', 'rho_b','dispersivity','lamb','alpha','kd'])
# convert back to real space
adv_trans['dispersivity'] = 10**adv_trans['dispersivity']
adv_trans['lamb'] = 10**adv_trans['lamb']

adv_trans['v'] = 1 # m/min
# calculate dispersion
adv_trans['D'] = adv_trans['v'] * adv_trans['dispersivity']
# calculate Peclet and Dahmkoler
adv_trans['Pe'] = (adv_trans['v'] * 2) / adv_trans['D']
adv_trans['Da'] = (adv_trans['lamb'] * 2) / adv_trans['v']

# dispersive reaction
problem = {
    'num_vars': 6,
    'names': ['theta', 'rho_b','dispersivity','lamb','alpha','kd'],
    'bounds': [[0.25, 0.7], # theta - porosity
               [0.29, 1.74], # rho_b - bulk density
               [np.log10(4), np.log10(1800)], # dispersivity
               [np.log10(8e-1), np.log10(500)], # lamb - first order decay rate constant
               [0, 0.01667], # alpha - first order desorption rate constant
               [0.01, 100]] # kd - sorption distribution coefficient
}

param_values = saltelli.sample(problem, 2**8)

disp_reaction = pd.DataFrame(data=param_values,
                         columns=['theta', 'rho_b','dispersivity','lamb','alpha','kd'])
# convert back to real space
disp_reaction['dispersivity'] = 10**disp_reaction['dispersivity']
disp_reaction['lamb'] = 10**disp_reaction['lamb']

disp_reaction['v'] = 1
# calculate dispersion
disp_reaction['D'] = disp_reaction['v'] * disp_reaction['dispersivity']
# calculate Peclet and Dahmkoler
disp_reaction['Pe'] = (disp_reaction['v'] * 2) / disp_reaction['D']
disp_reaction['Da'] = (disp_reaction['lamb'] * 2) / disp_reaction['v']


# dispersive transport
problem = {
    'num_vars': 6,
    'names': ['theta', 'rho_b','dispersivity','lamb','alpha','kd'],
    'bounds': [[0.25, 0.7], # theta - porosity
               [0.29, 1.74], # rho_b - bulk density
               [np.log10(4), np.log10(1800)], # dispersivity
               [np.log10(5e-4), np.log10(3e-1)], # lamb - first order decay rate constant
               [0, 24], # alpha - first order desorption rate constant
               [0.01, 100]] # kd - sorption distribution coefficient
}

param_values = saltelli.sample(problem, 2**8)

disp_trans = pd.DataFrame(data=param_values,
                         columns=['theta', 'rho_b','dispersivity','lamb','alpha','kd'])
# convert back to real space
disp_trans['dispersivity'] = 10**disp_trans['dispersivity']
disp_trans['lamb'] = 10**disp_trans['lamb']
disp_trans['v'] = 1
# calculate dispersion
disp_trans['D'] = disp_trans['v'] * disp_trans['dispersivity']
# calculate Peclet and Dahmkoler
disp_trans['Pe'] = (disp_trans['v'] * 2) / disp_trans['D']
disp_trans['Da'] = (disp_trans['lamb'] * 2) / disp_trans['v']


# plotting
colorblind_palette = sns.color_palette('colorblind', 4)

plt.figure(figsize=(8,8))
plt.scatter(disp_reaction['Pe'], disp_reaction['Da'], c=[colorblind_palette[0]], alpha=0.7, s = 5, label='Dispersion controlled reaction')
plt.scatter(adv_reaction['Pe'], adv_reaction['Da'], c=[colorblind_palette[1]], alpha=0.7, s=5, label='Advective controlled reaction')
plt.scatter(disp_trans['Pe'], disp_trans['Da'], c=[colorblind_palette[2]], alpha=0.7, s=5, label='Diffusive controlled transport')
plt.scatter(adv_trans['Pe'], adv_trans['Da'], c=[colorblind_palette[3]], alpha=0.7, s=5, label='Advective controlled transport')

# labeling
plt.annotate('Dispersion controlled reaction', (10**-1.8, 2700), fontweight='bold', fontsize=10, ha='center')
plt.annotate('Advective controlled reaction', (10**1.8, 2700), fontweight='bold', fontsize=10, ha='center')
plt.annotate('Dispersion controlled transport', (10**-1.8, 15**-3), fontweight='bold', fontsize=10, ha='center')
plt.annotate('Advective controlled transport', (10**1.8, 15**-3), fontweight='bold', fontsize=10, ha='center')

# formatting
plt.xlabel('Peclet Number', fontweight='bold')
plt.ylabel('Damkohler Number', fontweight='bold')
plt.xscale('log')
plt.yscale('log')
plt.xlim(1e-4,1e4)
plt.ylim(1e-4,1e4)
plt.axhline(1, c='black', linewidth=2)
plt.axvline(1, c='black', linewidth=2)
plt.grid(color='white', linestyle='--', linewidth=0.5)

fig_path = os.path.join(fig_dir, 'sampling_ranges.pdf')
plt.savefig(fig_path, format='pdf', bbox_inches='tight')
plt.show()


