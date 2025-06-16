import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import special
from scipy import interpolate
from numba import njit
from mpmath import invertlaplace
from mpmath import mp, exp
mp.dps = 12

def laplace_102(s, theta, rho_b, D, lamb, alpha, kd, Co, v, ts, L, x):
    '''Laplace time solution for a Type I boundary condition pulse injection in one dimension
    Returns a concentration value "C" in the laplace domain
    s: laplace frequency variable
    rho_b: bulk density
    D: dispersion
    lamb: first order decay rate constant
    alpha: first order desorption rate constant
    kd: sorption distribution coefficient
    Co: initial concentration (injected, not already present in system)
    v: pore velocity (now a static value between simulations)
    ts: pulse duration
    x: measured concentration location
    L: column length
    '''

    big_theta = s + lamb + (rho_b * alpha * kd * s) / (theta * (s + alpha))
    
    r1 = 1 / (2 * D) * (v + mp.sqrt(v ** 2 + 4 * D * big_theta))
    r2 = 1 / (2 * D) * (v - mp.sqrt(v ** 2 + 4 * D * big_theta))
    
    term1_numerator = r2 * mp.exp(r2 * L + r1 * x) - r1 * mp.exp(r1 * L + r2 * x)
    term1_denominator = r2 * mp.exp(r2 * L) - r1 * mp.exp(r1 * L)
    
    term1 = mp.fdiv(term1_numerator, term1_denominator)
    
    C = mp.fdiv(Co, s) * (1 - mp.exp(-ts * s)) * term1
    
    return C

def concentration_102_all_metrics(t, theta, rho_b, dispersivity, lamb, alpha, kd, Co, v, ts, L, x):
    '''Converts the laplace values from function laplace_102 to the real time domain
    Returns indexes for early arrival, peak concentration, and late time tailing, and an array of the concentration values
    Indexes are returned in dimensionless time
    '''
    concentration = []
    
    # convert to dimensionless time
    t = t/(L/v)

    D = v*dispersivity

    for time in t:
        if time == 0:
            conc = 0  # Assuming concentration at t=0 is Co 
        else:
            conc = invertlaplace(lambda s: laplace_102(s, theta, rho_b, D, lamb, alpha, kd, Co, v, ts, L, x), time, method='dehoog')
        concentration.append(conc)
    # Convert to array and normalize
    C_array = np.array(concentration, dtype=float) / Co
    
    # Find peak concentration
    peak_C = np.max(C_array)
    peak_index = np.argmax(C_array)

    # Compute 10% of peak concentration
    tenth_percentile_value = 0.1 * peak_C
    
    # Find the index where the concentration first reaches 10% of peak value
    early_arrival_idx = 0
    for i in range(len(C_array)):
        if C_array[i] >= tenth_percentile_value:
            early_arrival_idx = i
            break

    # Find the index where the concentration first reaches 10% of peak value
    late_arrival_idx = len(C_array)
    for i in range(peak_index, len(C_array)):
        if C_array[i] <= tenth_percentile_value:
            late_arrival_idx = i
            break

    return early_arrival_idx, peak_index, late_arrival_idx, C_array

def concentration_102_all_metrics_adaptive(t, theta, rho_b, dispersivity, lamb, alpha, kd, Co, v, ts, L, x):
    '''Converts the laplace solution from the function laplace_102 to the real time domain, with an adaptive time step to reduce computation time
    Returns indexes for early arrival, peak concentration, and late time tailing, and arrays of the concentration values and corresponding adaptive times
    Indexes are returned in dimensionless time
    '''
    # t is an input array of time values, the others are scalar parameters
    # initialize concentration and adaptive time lists
    concentration = []
    adaptive_times = []
    # convert to dimensionless time
    t = t/(L/v)
    
    D = v*dispersivity
    
    default_step = t.max()/len(t)
    current_time = 0
    
    # tolerance limit of step size
    tolerance = 0.01
    
    while current_time < t.max():
        if current_time == 0:
            conc = 0  # deal with time 0 case, if there is already concentration in the system change to that value
        else:
            conc = invertlaplace(lambda s: laplace_102(s, theta, rho_b, D, lamb, alpha, kd, Co, v, ts, L, x), current_time, method='dehoog')
        concentration.append(conc)
        adaptive_times.append(current_time)
        # check if concentration at current and previous time step changed substantially (> 1%)
        if len(concentration) < 2:
            current_time += default_step
        if len(concentration) > 1 and abs(concentration[-1] - concentration[-2]) > tolerance:
            current_time += default_step
        
        # speed up a lot if it's past the peak
        if len(concentration) > 1 and np.max(concentration) > 0 and concentration[-1] / np.max(concentration) < 0.1:
            current_time += default_step * 100
        else:
            current_time += default_step * 1.5
            
    # Convert to array and normalize
    C_array = np.array(concentration, dtype=float) / Co
    
    # Find peak concentration
    peak_C = np.max(C_array)
    peak_index = np.argmax(C_array)

    # Compute 10% of peak concentration
    tenth_percentile_value = 0.1 * peak_C
    
    # Find the index where the concentration first reaches 10% of peak value
    early_arrival_idx = 0
    for i in range(len(C_array)):
        if C_array[i] >= tenth_percentile_value:
            early_arrival_idx = i
            break

    # Find the index where the concentration first reaches 10% of peak value
    late_arrival_idx = len(C_array)
    for i in range(peak_index, len(C_array)):
        if C_array[i] <= tenth_percentile_value:
            late_arrival_idx = i
            break

    return early_arrival_idx, peak_index, late_arrival_idx, C_array, adaptive_times

def concentration_102_new_adaptive(t, theta, rho_b, dispersivity, lamb, alpha, kd, Co, v, ts, L, x):
    concentration = []
    adaptive_times = []
    
    # convert to dimensionless time
    t = t/(L/v)
    
    # calculate Dispersion
    D = v*dispersivity
    
    step_size = t.max()/len(t)
    tolerance = 0.10
    min_step = step_size * 1
    max_step = step_size * 100
    current_time = t[0]
    
    while current_time < t.max():
        if current_time == 0:
            conc = 0
        else:
            conc = invertlaplace(lambda s: laplace_102(s, theta, rho_b, D, lamb, alpha, kd, Co, v, ts, L, x), current_time, method='dehoog')

        concentration.append(conc)
        adaptive_times.append(current_time)
        
        # compute relative change in concentration
        if len(concentration) > 2:
            relative_change = abs((concentration[-1] - concentration[-2]) / concentration[-2])
        else:
            relative_change = 0
        
        # adjust step size based on relative change
        if relative_change > tolerance:
            step_size = max(min_step, step_size * 1)
        else:
            step_size = min(max_step, step_size * 2)
        
        current_time += step_size
        
        
    # Convert to array and normalize
    C_array = np.array(concentration, dtype=float) / Co
    
    return C_array, adaptive_times

def concentration_102_new_adaptive_extended(times, theta, rho_b, D, lamb, alpha, kd, Co, v, ts, L, x):
    """
    Simulates contaminant concentration with adaptive time-stepping 
    and dynamically extends the time domain if necessary.
    
    Parameters:
    - initial_times: array-like, initial time points
    - other parameters: model parameters
    
    Returns:
    - concentrations: list of concentration values
    - adaptive_times: list of time points used in the simulation
    """
    concentration = []
    adaptive_times = []
    
    # Convert to dimensionless time
    max_dimless_time = times.max() / (L/v)
    step_size = max_dimless_time / len(times)
    tolerance = 0.01
    min_step = step_size * 1
    max_step = step_size * 100
    current_time = times[0] / (L/v)
    
    while current_time < max_dimless_time:
        if current_time == 0:
            conc = 0
        else:
            conc = invertlaplace(lambda s: laplace_102(s, theta, rho_b, D, lamb, alpha, kd, Co, v, ts, L, x),
                                 current_time,  # Convert back to dimensional time
                                 method='dehoog')
        concentration.append(conc)
        adaptive_times.append(current_time)  # Convert back to dimensional time

        # Compute relative change in concentration
        if len(concentration) > 2:
            relative_change = abs((concentration[-1] - concentration[-2]) / max(concentration[-2], 1e-6))
        else:
            relative_change = 0

        # Adjust step size based on relative change
        if relative_change > tolerance:
            step_size = max(min_step, step_size * 1)
        else:
            step_size = min(max_step, step_size * 2)

        current_time += step_size

        # Stop condition: extend time domain if concentration hasn't dropped below 10% of peak
        if current_time > max_dimless_time and conc >= 0.1*max(concentration):
            max_dimless_time += max_dimless_time * 0.5
            #print('Extended!')
    
    C_array = np.array(concentration, dtype=float) / Co
    
    return C_array, adaptive_times


def laplace_106(s, theta, rho_b, D, lamb, alpha, kd, Co, v, ts, L, x):
    '''Laplace time solution for a Type III boundary condition pulse injection in one dimension
    Returns a concentration value "C" in the laplace domain
    s: laplace frequency variable
    rho_b: bulk density
    D: dispersion
    v: pore velocity
    lamb: first order decay rate constant
    alpha: first order desorption rate constant
    kd: sorption distribution coefficient
    Co: initial concentration (injected, not already present in system)
    ts: pulse duration
    x: measured concentration location
    L: column length
    '''

    big_theta = s + lamb + (rho_b * alpha * kd * s) / (theta * (s + alpha))
    delta = 1/(2*D) * mp.sqrt((v**2 + 4*D*big_theta))
    d = 2 * delta * L
    h = D/v
    sigma = v/(2*D)
    
    r1 = sigma + delta
    r2 = sigma - delta
    
    term1_numerator = r2 * mp.exp(r1 * x - d) - r1 * mp.exp(r2 * x)
    term1_denominator = r2 * (1 - h * r1) * mp.exp(-d) - (1 - h * r2)*r1
    
    term1 = mp.fdiv(term1_numerator, term1_denominator)
    
    C = mp.fdiv(Co, s) * (1 - mp.exp(-ts * s)) * term1
    
    return C

def concentration_106_all_metrics(t, theta, rho_b, dispersivity, lamb, alpha, kd, Co, v, ts, L, x):
    '''Converts the laplace values from function laplace_106 to the real time domain
    Returns indexes for early arrival, peak concentration, and late time tailing, and an array of the concentration values
    Indexes are returned in dimensionless time
    '''
    concentration = []
    
    # convert to dimensionless time
    t = t/(L/v)

    # calculate Dispersion
    D = v*dispersivity

    for time in t:
        if time == 0:
            conc = 0  # Assuming concentration at t=0 is Co 
        else:
            conc = invertlaplace(lambda s: laplace_106(s, theta, rho_b, D, v, lamb, alpha, kd, Co, v, ts, L, x), time, method='dehoog')
        concentration.append(conc)
    # Convert to array and normalize
    C_array = np.array(concentration, dtype=float) / Co
    
    # Find peak concentration
    peak_C = np.max(C_array)
    peak_index = np.argmax(C_array)

    # Compute 10% of peak concentration
    tenth_percentile_value = 0.1 * peak_C
    
    # Find the index where the concentration first reaches 10% of peak value
    early_arrival_idx = 0
    for i in range(len(C_array)):
        if C_array[i] >= tenth_percentile_value:
            early_arrival_idx = i
            break

    # Find the index where the concentration first reaches 10% of peak value
    late_arrival_idx = len(C_array)
    for i in range(peak_index, len(C_array)):
        if C_array[i] <= tenth_percentile_value:
            late_arrival_idx = i
            break

    return early_arrival_idx, peak_index, late_arrival_idx, C_array

def concentration_106_all_metrics_adaptive(t, theta, rho_b, dispersivity, lamb, alpha, kd, Co, v, ts, L, x):
    '''Converts the laplace solution from the function laplace_106 to the real time domain, with an adaptive time step to reduce computation time
    Returns indexes for early arrival, peak concentration, and late time tailing, and arrays of the concentration values and corresponding adaptive times
    Indexes are returned in dimensionless time
    '''
    # t is an input array of time values, the others are scalar parameters
    # initialize concentration and adaptive time lists
    concentration = []
    adaptive_times = []
    # convert to dimensionless time
    t = t/(L/v)
    # calculate Dispersion
    D = v*dispersivity
    
    default_step = t.max()/len(t)
    current_time = 0
    
    # tolerance limit of step size
    tolerance = 0.01
    
    while current_time < t.max():
        if current_time == 0:
            conc = 0  # deal with time 0 case, if there is already concentration in the system change to that value
        else:
            conc = invertlaplace(lambda s: laplace_106(s, theta, rho_b, D, lamb, alpha, kd, Co, v, ts, L, x), current_time, method='dehoog')
        concentration.append(conc)
        adaptive_times.append(current_time)
        # check if concentration at current and previous time step changed substantially (> 1%)
        if len(concentration) < 2:
            current_time += default_step
        if len(concentration) > 1 and abs(concentration[-1] - concentration[-2]) > tolerance:
            current_time += default_step
        
        # speed up a lot if it's past the peak
        if len(concentration) > 1 and np.max(concentration) > 0 and concentration[-1] / np.max(concentration) < 0.1:
            current_time += default_step * 100
        else:
            current_time += default_step * 1
            
    # Convert to array and normalize
    C_array = np.array(concentration, dtype=float) / Co
    
    # Find peak concentration
    peak_C = np.max(C_array)
    peak_index = np.argmax(C_array)

    # Compute 10% of peak concentration
    tenth_percentile_value = 0.1 * peak_C
    
    # Find the index where the concentration first reaches 10% of peak value
    early_arrival_idx = 0
    for i in range(len(C_array)):
        if C_array[i] >= tenth_percentile_value:
            early_arrival_idx = i
            break

    # Find the index where the concentration reaches the last 10% of peak value
    late_arrival_idx = len(C_array)
    for i in range(peak_index, len(C_array)):
        if C_array[i] <= tenth_percentile_value:
            late_arrival_idx = i
            break

    return early_arrival_idx, peak_index, late_arrival_idx, C_array, adaptive_times

def concentration_106_new_adaptive(t, theta, rho_b, dispersivity, lamb, alpha, kd, Co, v, ts, L, x):
    concentration = []
    adaptive_times = []
    
    # convert to dimensionless time
    t = t/(L/v)
    # calculate Dispersion
    D = v*dispersivity
    
    step_size = t.max()/len(t)
    tolerance = 0.10
    min_step = step_size * 1
    max_step = step_size * 100
    current_time = t[0]
    
    while current_time < t.max():
        if current_time == 0:
            conc = 0
        else:
            conc = invertlaplace(lambda s: laplace_106(s, theta, rho_b, D, lamb, alpha, kd, Co, v, ts, L, x), current_time, method='dehoog')

        concentration.append(conc)
        adaptive_times.append(current_time)
        
        # compute relative change in concentration
        if len(concentration) > 2:
            relative_change = abs((concentration[-1] - concentration[-2]) / concentration[-2])
        else:
            relative_change = 0
        
        # adjust step size based on relative change
        if relative_change > tolerance:
            step_size = max(min_step, step_size * 1)
        else:
            step_size = min(max_step, step_size * 2)
        
        current_time += step_size
        
        
    # Convert to array and normalize
    C_array = np.array(concentration, dtype=float) / Co
    
    return C_array, adaptive_times

def calculate_metrics(times, concentrations):
    """
    Extracts early arrival, peak concentration and late time tailing metrics from breakthrough curve data.  
    
    Parameters:
        - times: array-like, time values corresponding to the concentrations
        - concentrations: array-like, concentration values at each time step
    
    Returns:
        - early: time when concentration first reaches 10% of the peak
        - peak: time when peak concentration occurs
        - late: time when concentration drops below 10% of the peak
    """
    # find the early arrival time
    peak_value = np.max(concentrations)
    tenth_percentile_value = 0.1 * peak_value
    early_idx = 0
    for i in range(len(concentrations)):
        if concentrations[i] >= tenth_percentile_value:
            early_idx = i
            break
    early = times[early_idx]

    # find the peak index
    peak_idx = np.argmax(concentrations)
    peak = times[peak_idx]
    
    # find the late time tailing index
    late_idx = len(times) - 1
    for i in range(peak_idx, len(concentrations)):
        if concentrations[i] <= tenth_percentile_value:
            late_idx = i
            break
    late = times[late_idx]
    
    return early, peak, late

def concentration_106_new_adaptive_extended(times, theta, rho_b, D, lamb, alpha, kd, Co, v, ts, L, x):
    """
    Simulates contaminant concentration with adaptive time-stepping 
    and dynamically extends the time domain if necessary.
    
    Parameters:
    - initial_times: array-like, initial time points
    - other parameters: model parameters
    
    Returns:
    - concentrations: list of concentration values
    - adaptive_times: list of time points used in the simulation
    """
    concentration = []
    adaptive_times = []
    
    # Convert to dimensionless time
    max_dimless_time = times.max() / (L/v)
    step_size = max_dimless_time / len(times)
    tolerance = 0.01
    min_step = step_size * 1
    max_step = step_size * 100
    current_time = times[0] / (L/v)
    
    while current_time < max_dimless_time:
        if current_time == 0:
            conc = 0
        else:
            conc = invertlaplace(lambda s: laplace_106(s, theta, rho_b, D, lamb, alpha, kd, Co, v, ts, L, x),
                                 current_time,  # Convert back to dimensional time
                                 method='dehoog')
        concentration.append(conc)
        adaptive_times.append(current_time)  # Convert back to dimensional time

        # Compute relative change in concentration
        if len(concentration) > 2:
            relative_change = abs((concentration[-1] - concentration[-2]) / max(concentration[-2], 1e-6))
        else:
            relative_change = 0

        # Adjust step size based on relative change
        if relative_change > tolerance:
            step_size = max(min_step, step_size * 1)
        else:
            step_size = min(max_step, step_size * 2)

        current_time += step_size

        # Stop condition: extend time domain if concentration hasn't dropped below 10% of peak
        if current_time > max_dimless_time and conc >= 0.1*max(concentration):
            max_dimless_time += max_dimless_time * 0.5
            #print('Extended!')
    
    C_array = np.array(concentration, dtype=float) / Co
    #C_array = np.array(concentration, dtype=float)
    
    return C_array, adaptive_times

#--------------------------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------------------------
# below are draft or formerly used equations that are no longer part of the main analysis

def concentration_102(t, theta, rho_b, D, v, lamb, alpha, kd, Co, L):
    # convert the input times to dimensionless time
    t = t/(L/v)
    concentration = []

    for time in t:
        if time == 0:
            conc = 0
        else:
            conc = invertlaplace(lambda s: laplace_102(s, theta, rho_b, D, v, lamb, alpha, kd, Co, ts=5, x=2, L=2), time, method='dehoog')
        concentration.append(conc)
    print('transformed')
    return concentration

def concentration_102_early_arrival(t, theta, rho_b, D, v, lamb, alpha, kd, Co, L):
    # Compute concentration for each time t
    concentration = []
    
    # convert to dimensionless time
    t = t/(L/v)

    for time in t:
        if time == 0:
            conc = 0  # Assuming concentration at t=0 is Co (you may adjust as needed)
        else:
            conc = invertlaplace(lambda s: laplace_102(s, theta, rho_b, D, v, lamb, alpha, kd, Co, ts=5, x=2, L=2), time, method='dehoog')
        concentration.append(conc)
    # Convert to array and normalize
    C_array = np.array(concentration, dtype=float) / Co
    
    # Find peak concentration
    peak_C = np.max(C_array)
    peak_index = np.argmax(C_array)

    # Compute 10% of peak concentration
    tenth_percentile_value = 0.1 * peak_C

    # Find the index where the concentration first reaches 10% of peak value
    for i in range(len(C_array)):
        if C_array[i] >= tenth_percentile_value:
            early_arrival_idx = i
            break
    
    # Find the corresponding time value
    early_arrival = t[early_arrival_idx]

    return early_arrival_idx

def concentration_102_peak(t, theta, rho_b, D, v, lamb, alpha, kd, Co, L):
    # convert to dimensionless time
    t = t/(L/v)

    # Compute concentration for each time t
    concentration = []
    for time in t:
        if time == 0:
            conc = 0  # Assuming concentration at t=0 is Co (you may adjust as needed)
        else:
            conc = invertlaplace(lambda s: laplace_102(s, theta, rho_b, D, v, lamb, alpha, kd, Co, ts=5, x=2, L=2), time, method='dehoog')
        concentration.append(conc)
    # Convert to array and to dimensionless concentration
    C_array = np.array(concentration, dtype=float) / Co
    
    # Find peak concentration
    peak_C = np.max(C_array)
    peak_index = np.argmax(C_array)
    
    # Find the corresponding time value
    peak_time = t[peak_index]

    return peak_index

def concentration_102_late_arrival(t, theta, rho_b, D, v, lamb, alpha, kd, Co, L):
    # Compute concentration for each time t
    concentration = []
    
    # convert to dimensionless time
    t = t/(L/v)

    for time in t:
        if time == 0:
            conc = 0  # Assuming concentration at t=0 is Co (you may adjust as needed)
        else:
            conc = invertlaplace(lambda s: laplace_102(s, theta, rho_b, D, v, lamb, alpha, kd, Co, ts=5, x=2, L=2), time, method='dehoog')
        concentration.append(conc)
    # Convert to array and normalize
    C_array = np.array(concentration, dtype=float) / Co
    
    # Find peak concentration
    peak_C = np.max(C_array)
    peak_index = np.argmax(C_array)

    # Compute 10% of peak concentration
    tenth_percentile_value = 0.1 * peak_C
    late_arrival_idx = 0
    # Find the index where the concentration first reaches 10% of peak value
    for i in range(peak_index, len(C_array)):
        if C_array[i] <= tenth_percentile_value:
            late_arrival_idx = i
            break

    return late_arrival_idx


def concentration_106(t, theta, rho_b, D, v, lamb, alpha, kd, Co):
    concentration = [invertlaplace(lambda s: laplace_106(s, theta, rho_b, D, v, lamb, alpha, kd, Co, ts=5, x=2, L=2), time, method='dehoog') for time in t]
    print('transformed')
    return concentration

def concentration_106_early_arrival(t, theta, rho_b, D, v, lamb, alpha, kd, Co):
    # Compute concentration for each time t
    concentration = []
    for time in t:
        if time == 0:
            conc = 0  # Assuming concentration at t=0 is Co (you may adjust as needed)
        else:
            conc = invertlaplace(lambda s: laplace_106(s, theta, rho_b, D, v, lamb, alpha, kd, Co, ts=5, x=2, L=2), time, method='dehoog')
        concentration.append(conc)
    # Convert to array and normalize
    C_array = np.array(concentration, dtype=float) / Co
    
    # Find peak concentration
    peak_C = np.max(C_array)
    peak_index = np.argmax(C_array)

    # Compute 10% of peak concentration
    tenth_percentile_value = 0.1 * peak_C

    # Find the index where the concentration first reaches 10% of peak value
    for i in range(len(C_array)):
        if C_array[i] >= tenth_percentile_value:
            early_arrival_idx = i
            break
    
    # Find the corresponding time value
    early_arrival = t[early_arrival_idx]

    return early_arrival


# define the analytical solution for concentration in a type 1 boundary condition environment
# base analytical solution
def bc1(t, Co, q, p, R):
    
    diffusion = 1.0 * 10**-9
    v = q / p / R
    alpha = 0.83 * np.log10(L)**2.414
    Dl = (alpha * v + diffusion) / R

    first_term = special.erfc((L - v * t) / (2 * np.sqrt(Dl * t)))
    second_term = np.exp(v * L / Dl) * special.erfc((L + v * t) / (2 * np.sqrt(Dl * t)))

    C = (Co / 2) * (first_term + second_term)

    return C



def first_arrival_bc1(t, Co, q, p, R): # t is an array of input times
    
    # check if t is a single number or an iterable
    if np.isscalar(t):
        t = np.array([t])
    
    concs = np.array([bc1(time, Co, q, p, R) for time in t])
    peak = np.max(concs)
    arrival_indices = np.where(concs >= 0.10 * peak)[0]
    
    if arrival_indices.size > 0:
        return t[arrival_indices[0]]
    else:
        return np.nan


# peak concentration
def peak_conc_bc1(t, Co, q, p, R):
    
    time = np.arange(0,t,1)
    concs = bc1(time, Co, q, p, R)
    peak = np.max(concs)
        
    return peak


# define the analytical solution for concentration in a type 2 boundary condition environment
def bc2(t, Co, q, p, R):
    diffusion = 1.0 * 10**-9
    v = q / p / R
    alpha = 0.83 * np.log10(L)**2.414
    Dl = (alpha * v + diffusion) / R

    first_term = special.erfc((L - v * t) / (2 * np.sqrt(Dl * t)))
    second_term = np.exp(v * L / Dl) * special.erfc((L + v * t) / (2 * np.sqrt(Dl * t)))

    C = (Co / 2) * (first_term - second_term)

    return C

# first arrival bc2
def first_arrival_bc2(t, Co, q, p, R):
    
    time = np.arange(0,t,1)
    concs = bc2(time, Co, q, p, R)
    peak = np.max(concs)
    arrival_indices = np.where(concs >= 0.10 * peak)[0]
    
    if arrival_indices.size > 0:
        return time[arrival_indices[0]]
    else:
        return None

# peak concentration bc2
def peak_conc_bc2(t, Co, q, p, R):
    
    time = np.arange(0,t,1)
    concs = bc2(time, Co, q, p, R)
    peak = np.max(concs)
    peak_indices = np.where(concs >= 0.10 * peak)[0]
    
    if peak_indices.size > 0:
        return time[peak_indices[0]]
    else:
        return None

# define the analytical solution for concentration in a type 3 boundary condition environment
def bc3(t, Co, q, p, R):
    diffusion = 1.0 * 10**-9
    v = q / p / R
    alpha = 0.83 * np.log10(L)**2.414
    Dl = (alpha * v + diffusion) / R

    first_term = special.erfc((L - v * t)/(2 * np.sqrt(Dl * t))) + np.sqrt((v**2 * t)/(np.pi*Dl)) * np.exp(-(L - v *t)**2/(4*Dl*t))
    second_term = 0.5*(1 + v*L/Dl + v**2*t/Dl) * np.exp(v*L/Dl) * special.erfc((L + v *t)/(2 * np.sqrt(Dl * t)))

    C = (Co/2) * (first_term - second_term)

    return C

# first arrival bc3
def first_arrival_bc3(t, Co, q, p, R):
    
    time = np.arange(0,t,1)
    concs = bc3(time, Co, q, p, R)
    peak = np.max(concs)
    arrival_indices = np.where(concs >= 0.10 * peak)[0]
    
    if arrival_indices.size > 0:
        return time[arrival_indices[0]]
    else:
        return None

# peak concentration bc3
def peak_conc_bc3(t, Co, q, p, R):
    
    time = np.arange(0,t,1)
    concs = bc3(time, Co, q, p, R)
    peak = np.max(concs)
    peak_indices = np.where(concs >= 0.10 * peak)[0]
    
    if peak_indices.size > 0:
        return time[peak_indices[0]]
    else:
        return None

# note: bc3 differs slightly from Veronica's notes. Van Genuchten describes the second term with v**2*t/Dl, Veron had v**2*L/Dl
# also VG had special.erfc(L + v *t....etc) while V had special.erfc(L - v * t....etc) I was getting negative concentration values with V's version




#%% one dimensional first-type finite pulse BC (concentration Co for duration t)

def continuous_bc1(v,lamb,Dx,x,t,R,Ci):
    # v = velocity?
    # lamb(lambda) = first order rate constant
    # Dx = dispersion coefficient
    # x = position
    # t = time
    # R = retardation factor
    # Ci = initial concentration
    
    u = v*np.sqrt(1+(4*lamb*Dx/v**2))
    
    first_part = np.exp((v-u)*x/(2*Dx)) * special.erfc((R*x - u*t)/(2*np.sqrt(Dx*R*t))) + \
                 np.exp((v+u)*x/(2*Dx)) * special.erfc((R*x + u*t)/(2*np.sqrt(Dx*R*t)))
    
    second_part = special.erfc((R*x - v*t)/(2*np.sqrt(Dx*R*t))) + \
                  np.exp(v*x/Dx)*special.erfc((R*x + v*t)/(2*np.sqrt(Dx*R*t)))
    
    C = 0.5*Co*first_part - 0.5*Ci*np.exp(lamb*t/R)*second_part + Ci*np.exp(-lamb*t/R)
    
    return C, first_part, second_part



# v = 0.01
# lamb = 0.5
# Dx = 1.0**10-9
# t = np.arange(0,1000,1)
# R = 1


# test, first, second = continuous_bc1(v,lamb,Dx,2,t,R,0)
# plt.plot(test)



#%%
def concentration(x, t, C0, u, v, Dx, R, lt, Ci):
    term1 = (v - u) * x / (2 * Dx)
    term2 = special.erfc((R * x - u * t) / (2 * np.sqrt(Dx * R * t)))
    term3 = np.exp((v + u) * x / (2 * Dx)) * special.erfc((R * x + u * t) / (2 * np.sqrt(Dx * R * t)))
    term4 = np.exp(v * x / Dx) * special.erfc((R * x + v * t) / (2 * np.sqrt(Dx * R * t)))
    
    return 0.5 * C0 * (np.exp(term1) * term2 + term3) - 0.5 * Ci * np.exp(-lt * t / R) * term4 + Ci * np.exp(-lt * t / R)


#%% continuous injection Goltz page 37
def continuous(x, t, Co, v, R, lamb, Ci):
    
    diffusion = 1.0 * 10**-9 # diffusion constant
    alpha = 0.83 * np.log10(x)**2.414 # dispersivity
    Dx = (alpha * v + diffusion) / R # dispersion coefficient
    u = v*np.sqrt(1+(4*lamb*Dx)/v**2)

    term1 = np.exp((v - u) * x / (2 * Dx))
    term2 = special.erfc((R * x - u * t) / (2 * np.sqrt(Dx * R * t)))
    term3 = np.exp((v + u) * x / (2 * Dx))
    term4 = special.erfc((R * x + u * t) / (2 * np.sqrt(Dx * R * t)))
    
    term5 = special.erfc((R * x - v * t) / (2 * np.sqrt(Dx * R * t)))
    term6 = np.exp(v*x/Dx)
    term7 = special.erfc((R * x + v * t) / (2 * np.sqrt(Dx * R * t)))

    return 0.5*Co * (term1*term2+term3*term4) - 0.5*Ci*np.exp(-lamb*t/R)*(term5+term6*term7) + Ci*np.exp(-lamb*t/R)


#%% pulse injection Goltz page 39

def pulse(x, t_scalar, ts, Co, v, R, lamb, Ci):
    
    #diffusion = 1.0 * 10 ** -9  # diffusion constant
    diffusion = 0
    #alpha = 0.83 * np.log10(x) ** 2.414
    #alpha = 0.1
    #Dx = (alpha * v + diffusion) / R  # dispersion coefficient
    Dx = 1000
    u = v * np.sqrt(1 + (4 * lamb * Dx) / v ** 2)
    
    term1 = np.exp((v - u) * x / (2 * Dx))
    term2 = special.erfc((R * x - u * t_scalar) / (2 * np.sqrt(Dx * R * t_scalar)))
    term3 = np.exp((v + u) * x / (2 * Dx))
    term4 = special.erfc((R * x + u * t_scalar) / (2 * np.sqrt(Dx * R * t_scalar)))

    term5 = special.erfc((R * x - v * t_scalar) / (2 * np.sqrt(Dx * R * t_scalar)))
    term6 = np.exp(v * x / Dx)
    term7 = special.erfc((R * x + v * t_scalar) / (2 * np.sqrt(Dx * R * t_scalar)))
    
    # Ensure that we don't compute terms that involve negative square roots
    if t_scalar < ts:
        C = 0.5 * Co * (term1 * term2 + term3 * term4) - 0.5 * Ci * np.exp(-lamb * t_scalar / R) * (term5 + term6 * term7) + Ci * np.exp(-lamb * t_scalar / R)
        print('use first time at time: '+str(t_scalar))
    else:
        term8 = special.erfc((R * x - u * (t_scalar - ts)) / (2 * np.sqrt(Dx * R * (t_scalar - ts))))
        term9 = special.erfc((R * x + u * (t_scalar - ts)) / (2 * np.sqrt(Dx * R * (t_scalar - ts))))
        C = 0.5 * Co * (term1 * term2 + term3 * term4) - 0.5 * Ci * np.exp(-lamb * t_scalar / R) * (term5 + term6 * term7) + Ci * np.exp(-lamb * t_scalar / R) - \
            0.5 * Co * (term1 * term8 + term3 * term9)
        print('use second term at time: '+str(t_scalar))
    
    return C,Dx,u


#%% Van Genuchten solution

from scipy.special import erfc

def A(x, t, v, D, R):
    if t > 0:
        term1 = erfc((R * x - v * t) / (2 * np.sqrt(D * R * t)))
        term2 = np.sqrt(v**2*t/(np.pi*D*R)) * np.exp(-(R*x - v*t)**2/(4*D*R*t))
        term3 = (1 + v*x/D + v**2*t/(D*R)) * np.exp(v*x/D) * erfc((R*x + v*t)/(2*np.sqrt(D*R*t)))
        return 0.5*term1 + term2 - 0.5*term3
    else:
        return 0
    
def B(x, t, v, D, R,gamma):
    if t > 0:
        term1 = t + 1/(2*v) * (1+ v*x/D + v**2*t/(D*R)) * erfc((R * x + v * t) / (2 * np.sqrt(D * R * t)))
        term2 = np.sqrt(t/(4*np.pi*D*R)) * (R*x + v*t + 2*D*R/v) * erfc((R * x - v * t) / (2 * np.sqrt(D * R * t)))
        term3 = (t/2 - (D*R)/(2*v**2)) * np.exp(v*x/D) * erfc((R * x + v * t) / (2 * np.sqrt(D * R * t)))
        return gamma / R * (term1 - term2 + term3)
    else:
        return 0
    

def pulse_concentration(x, t, Co, Ci, to, v, R, D, gamma):
    if t < to:
        C = Ci + (Co - Ci) * A(x, t, v, D, R) + B(x, t, v, D, R, gamma)
        return C
    if t == to:
        C = Ci + (Co - Ci) * A(x, t, v, D, R) + B(x, t, v, D, R, gamma)
        return C
    else:
        C = Ci + (Co - Ci) * A(x, t, v, D, R) + B(x, t, v, D, R, gamma) - Co * A(x, (t - to), v, D, R)
        return C




