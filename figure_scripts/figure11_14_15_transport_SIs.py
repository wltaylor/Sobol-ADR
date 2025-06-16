#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  4 14:56:37 2025

@author: williamtaylor
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json
import matplotlib.patches as mpatches
plt.style.use('default')

base_dir = os.path.dirname(os.path.abspath(__file__))
results_dir = os.path.normpath(os.path.join(base_dir, '..', 'results'))
fig_dir = os.path.normpath(os.path.join(base_dir, '..', 'figures'))

# helper functions
def read_result_csv(filename):
    return pd.read_csv(os.path.join(results_dir, filename), index_col=0)

def read_result_json(filename):
    with open(os.path.join(results_dir, filename), 'r') as f:
        data = json.load(f)
    return data

transports = ['dr','ar','dt','at']
SIs_dict = {}
for transport in transports:
    total_Si_early = read_result_csv(f'total_Si_early_{transport}.csv')
    first_Si_early = read_result_csv(f'first_Si_early_{transport}.csv')
    second_Si_early = read_result_csv(f'second_Si_early_{transport}.csv')
    total_Si_peak = read_result_csv(f'total_Si_peak_{transport}.csv')
    first_Si_peak = read_result_csv(f'first_Si_peak_{transport}.csv')
    second_Si_peak = read_result_csv(f'second_Si_peak_{transport}.csv')
    total_Si_late = read_result_csv(f'total_Si_late_{transport}.csv')
    first_Si_late = read_result_csv(f'first_Si_late_{transport}.csv')
    second_Si_late = read_result_csv(f'second_Si_late_{transport}.csv')
    
    SIs_dict[transport] = {
        'total_Si_early': total_Si_early,
        'first_Si_early': first_Si_early,
        'second_Si_early': second_Si_early,
        'total_Si_peak': total_Si_peak,
        'first_Si_peak': first_Si_peak,
        'second_Si_peak': second_Si_peak,
        'total_Si_late': total_Si_late,
        'first_Si_late': first_Si_late,
        'second_Si_late': second_Si_late
        }
    indices = ['early','peak','late']
    types = ['total','first','second']
    names= ['theta', 'rho_b','alpha','lamb','dispersivity','kd']
    

scenarios = ['dr','ar','dt','at']
titles = ['a. Dispersive Reaction (Pe<1, Da>1)','b. Advective Reaction (Pe>1, Da>1)','c. Dispersive Transport (Pe<1, Da<1)', 'd. Advective Transport (Pe>1, Da<1)']
names = ['theta','rho_b','alpha','lamb','dispersivity','kd']
greek_labels = [r'$\theta$', r'$\rho_b$', r'$\alpha$', r'$\lambda$', r'$\alpha_i$', r'$K_d$']
positions = np.arange(1,len(names) + 1, 1)
time = 'early'
fig, axes = plt.subplots(2,2,figsize=(8,8))
axes = axes.flatten()
for j,scenario in enumerate(scenarios):
    st_values = {name: SIs_dict[scenario]['total_Si_'+str(time)]['ST'][name] for name in names}
    threshold = 0.25*np.max(list(st_values.values()))
    for i, name in enumerate(names):
       if st_values[name] < threshold:    
           axes[j].barh(positions[i], SIs_dict[scenario]['total_Si_'+str(time)]['ST'][name], height=0.3, color='red', edgecolor='black', alpha=0.33)
           axes[j].errorbar(SIs_dict[scenario]['total_Si_'+str(time)]['ST'][name], positions[i], 
                     xerr=SIs_dict[scenario]['total_Si_'+str(time)]['ST_conf'][name], fmt='none', ecolor='black', capsize=2, alpha=0.33)
       else:
           axes[j].barh(positions[i], SIs_dict[scenario]['total_Si_'+str(time)]['ST'][name], height=0.3, color='red', edgecolor='black', alpha=1)
           axes[j].errorbar(SIs_dict[scenario]['total_Si_'+str(time)]['ST'][name], positions[i], 
                     xerr=SIs_dict[scenario]['total_Si_'+str(time)]['ST_conf'][name], fmt='none', ecolor='black', capsize=2, alpha=1)
       # First order
       if st_values[name] < threshold:
           axes[j].barh(positions[i] - 0.3, SIs_dict[scenario]['first_Si_'+str(time)]['S1'][name], color='blue', height=0.3, edgecolor='black', alpha=0.33)
       else:
           axes[j].barh(positions[i] - 0.3, SIs_dict[scenario]['first_Si_'+str(time)]['S1'][name], color='blue', height=0.3, edgecolor='black', alpha=1)

       # Second order
       second_order_contributions = SIs_dict[scenario]['second_Si_'+str(time)]['S2']
       second_order_conf_ints = SIs_dict[scenario]['second_Si_'+str(time)]['S2_conf']
       second_order_sum = second_order_contributions.loc[second_order_contributions.index.map(lambda x: name in x)].sum()
       
       bottom_position = SIs_dict[scenario]['first_Si_'+str(time)]['S1'][name]

       if st_values[name] < threshold:  
           axes[j].barh(positions[i] - 0.3, 
                     second_order_sum,
                     left=bottom_position,
                     color='skyblue',
                     height=0.3,
                     edgecolor='black',
                     alpha=0.33)
       else:
           axes[j].barh(positions[i] - 0.3, 
                     second_order_sum,
                     left=bottom_position,
                     color='skyblue',
                     height=0.3,
                     edgecolor='black',
                     alpha=1)
           
    axes[j].axvline(threshold, c='black', linestyle='--')
    axes[j].set_xlim(0,1.2)
    axes[j].set_title(titles[j], fontweight='bold', fontsize=12)
    axes[j].set_yticks(positions)
    axes[j].set_yticklabels(greek_labels)

handles = [
    mpatches.Patch(color='red', label='Total Order'),
    mpatches.Patch(color='blue', label='First Order'),
    mpatches.Patch(color='skyblue', label='Second Order')
]
fig.legend(handles=handles, loc='center', edgecolor='black', ncol=3, bbox_to_anchor=(0.52, -0.01))
plt.tight_layout()
fig_path = os.path.join(fig_dir, f'{time}_transport_SIs_quad.pdf')
plt.savefig(fig_path, format='pdf', bbox_inches='tight')
plt.show()

#----------------------------------------------
#----------------------------------------------
# time to peak concentration
time = 'peak'
fig, axes = plt.subplots(2,2,figsize=(8,8))
axes = axes.flatten()
for j,scenario in enumerate(scenarios):
    st_values = {name: SIs_dict[scenario]['total_Si_'+str(time)]['ST'][name] for name in names}
    threshold = 0.25*np.max(list(st_values.values()))
    for i, name in enumerate(names):
       if st_values[name] < threshold:    
           axes[j].barh(positions[i], SIs_dict[scenario]['total_Si_'+str(time)]['ST'][name], height=0.3, color='red', edgecolor='black', alpha=0.33)
           axes[j].errorbar(SIs_dict[scenario]['total_Si_'+str(time)]['ST'][name], positions[i], 
                     xerr=SIs_dict[scenario]['total_Si_'+str(time)]['ST_conf'][name], fmt='none', ecolor='black', capsize=2, alpha=0.33)
       else:
           axes[j].barh(positions[i], SIs_dict[scenario]['total_Si_'+str(time)]['ST'][name], height=0.3, color='red', edgecolor='black', alpha=1)
           axes[j].errorbar(SIs_dict[scenario]['total_Si_'+str(time)]['ST'][name], positions[i], 
                     xerr=SIs_dict[scenario]['total_Si_'+str(time)]['ST_conf'][name], fmt='none', ecolor='black', capsize=2, alpha=1)
       # First order
       if st_values[name] < threshold:
           axes[j].barh(positions[i] - 0.3, SIs_dict[scenario]['first_Si_'+str(time)]['S1'][name], color='blue', height=0.3, edgecolor='black', alpha=0.33)
       else:
           axes[j].barh(positions[i] - 0.3, SIs_dict[scenario]['first_Si_'+str(time)]['S1'][name], color='blue', height=0.3, edgecolor='black', alpha=1)

       # Second order
       second_order_contributions = SIs_dict[scenario]['second_Si_'+str(time)]['S2']
       second_order_conf_ints = SIs_dict[scenario]['second_Si_'+str(time)]['S2_conf']
       second_order_sum = second_order_contributions.loc[second_order_contributions.index.map(lambda x: name in x)].sum()
       
       bottom_position = SIs_dict[scenario]['first_Si_'+str(time)]['S1'][name]

       if st_values[name] < threshold:  
           axes[j].barh(positions[i] - 0.3, 
                     second_order_sum,
                     left=bottom_position,
                     color='skyblue',
                     height=0.3,
                     edgecolor='black',
                     alpha=0.33)
       else:
           axes[j].barh(positions[i] - 0.3, 
                     second_order_sum,
                     left=bottom_position,
                     color='skyblue',
                     height=0.3,
                     edgecolor='black',
                     alpha=1)
           
    axes[j].axvline(threshold, c='black', linestyle='--')
    axes[j].set_xlim(0,1.2)
    axes[j].set_title(titles[j], fontweight='bold', fontsize=12)
    axes[j].set_yticks(positions)
    axes[j].set_yticklabels(greek_labels)

handles = [
    mpatches.Patch(color='red', label='Total Order'),
    mpatches.Patch(color='blue', label='First Order'),
    mpatches.Patch(color='skyblue', label='Second Order')
]
fig.legend(handles=handles, loc='center', edgecolor='black', ncol=3, bbox_to_anchor=(0.52, -0.01))
plt.tight_layout()
fig_path = os.path.join(fig_dir, f'{time}_transport_SIs_quad.pdf')
plt.savefig(fig_path, format='pdf', bbox_inches='tight')
plt.show()

#----------------------------------------------
#----------------------------------------------
# late time tailing
time = 'late'
fig, axes = plt.subplots(2,2,figsize=(8,8))
axes = axes.flatten()
for j,scenario in enumerate(scenarios):
    st_values = {name: SIs_dict[scenario]['total_Si_'+str(time)]['ST'][name] for name in names}
    threshold = 0.25*np.max(list(st_values.values()))
    for i, name in enumerate(names):
       if st_values[name] < threshold:    
           axes[j].barh(positions[i], SIs_dict[scenario]['total_Si_'+str(time)]['ST'][name], height=0.3, color='red', edgecolor='black', alpha=0.33)
           axes[j].errorbar(SIs_dict[scenario]['total_Si_'+str(time)]['ST'][name], positions[i], 
                     xerr=SIs_dict[scenario]['total_Si_'+str(time)]['ST_conf'][name], fmt='none', ecolor='black', capsize=2, alpha=0.33)
       else:
           axes[j].barh(positions[i], SIs_dict[scenario]['total_Si_'+str(time)]['ST'][name], height=0.3, color='red', edgecolor='black', alpha=1)
           axes[j].errorbar(SIs_dict[scenario]['total_Si_'+str(time)]['ST'][name], positions[i], 
                     xerr=SIs_dict[scenario]['total_Si_'+str(time)]['ST_conf'][name], fmt='none', ecolor='black', capsize=2, alpha=1)
       # First order
       if st_values[name] < threshold:
           axes[j].barh(positions[i] - 0.3, SIs_dict[scenario]['first_Si_'+str(time)]['S1'][name], color='blue', height=0.3, edgecolor='black', alpha=0.33)
       else:
           axes[j].barh(positions[i] - 0.3, SIs_dict[scenario]['first_Si_'+str(time)]['S1'][name], color='blue', height=0.3, edgecolor='black', alpha=1)

       # Second order
       second_order_contributions = SIs_dict[scenario]['second_Si_'+str(time)]['S2']
       second_order_conf_ints = SIs_dict[scenario]['second_Si_'+str(time)]['S2_conf']
       second_order_sum = second_order_contributions.loc[second_order_contributions.index.map(lambda x: name in x)].sum()
       
       bottom_position = SIs_dict[scenario]['first_Si_'+str(time)]['S1'][name]

       if st_values[name] < threshold:  
           axes[j].barh(positions[i] - 0.3, 
                     second_order_sum,
                     left=bottom_position,
                     color='skyblue',
                     height=0.3,
                     edgecolor='black',
                     alpha=0.33)
       else:
           axes[j].barh(positions[i] - 0.3, 
                     second_order_sum,
                     left=bottom_position,
                     color='skyblue',
                     height=0.3,
                     edgecolor='black',
                     alpha=1)
           
    axes[j].axvline(threshold, c='black', linestyle='--')
    axes[j].set_xlim(0,1.2)
    axes[j].set_title(titles[j], fontweight='bold', fontsize=12)
    axes[j].set_yticks(positions)
    axes[j].set_yticklabels(greek_labels)

handles = [
    mpatches.Patch(color='red', label='Total Order'),
    mpatches.Patch(color='blue', label='First Order'),
    mpatches.Patch(color='skyblue', label='Second Order')
]
fig.legend(handles=handles, loc='center', edgecolor='black', ncol=3, bbox_to_anchor=(0.52, -0.01))
plt.tight_layout()
fig_path = os.path.join(fig_dir, f'{time}_transport_SIs_quad.pdf')
plt.savefig(fig_path, format='pdf', bbox_inches='tight')
plt.show()
