from brian2 import *
import importlib
import json
import numpy as np 
from brian2 import mV, ms, nS, Hz
import gc  

def load_params_from_file(params_file):
    try:
        with open(params_file, 'r') as f:
            data = json.load(f)
            return data['params']
    except Exception as e:
        print(f"Load Error: {str(e)}")
        raise

def get_neuron_count(neuron_configs, target_name):
    for config in neuron_configs:
        if config["name"] == target_name:
            return config["N"]
    return None  

def create_poisson_input(N, rate_expr, duration, dt=1*ms):
    time_points = np.arange(0, duration/ms, dt/ms) * ms
    chunk_size = 1000
    total_chunks = int(np.ceil(len(time_points) / chunk_size))
    rate_arrays = []
    
    for i in range(total_chunks):
        start_idx = i * chunk_size
        end_idx = min((i + 1) * chunk_size, len(time_points))
        chunk_times = time_points[start_idx:end_idx]
        chunk_rates = np.array([
            eval(rate_expr, {"t": t, "Hz": Hz, "ms": ms, "second": second, "np": np})
            for t in chunk_times
        ])
        rate_arrays.append(chunk_rates)
    
    rate_values = np.concatenate(rate_arrays)
    per_neuron_rate_values = rate_values / N
    rate_array = TimedArray(per_neuron_rate_values * Hz, dt)
    
    del rate_arrays, rate_values, per_neuron_rate_values
    gc.collect()
    
    return PoissonGroup(N, rates='rate_array(t)', namespace={'rate_array': rate_array})

def generate_non_overlapping_poisson_input(N, rate, duration, dt=1):

    n_steps = int(np.ceil(duration / dt))
    
    indices_list = []
    times_list = []
    
    for step in range(n_steps):
        t = step * dt
        n_spikes = np.random.poisson(rate * dt)
        if n_spikes > 0:
            n_spikes = min(n_spikes, N)
            chosen = np.random.choice(N, size=n_spikes, replace=False)
            for i in chosen:
                indices_list.append(i)
                times_list.append(t)
    
    if not indices_list:
        return np.array([], dtype=int), np.array([], dtype=float)
    
    indices = np.array(indices_list, dtype=int)
    times = np.array(times_list, dtype=float)
    
    order = np.lexsort((indices, times))
    indices = indices[order]
    times = times[order]
    
    return indices, times

def analyze_input_requirements(name, total_rate, N, target_min_hz, target_max_hz):
    current_input_per_neuron = total_rate / N
    target_mean = (target_min_hz + target_max_hz) / 2
    ratio = target_mean / current_input_per_neuron if current_input_per_neuron > 0 else float('inf')
    
    print(f"\nInput Analysis for {name}:")
    print(f"  Current input per neuron: {current_input_per_neuron:.2f} Hz")
    print(f"  Required scaling factor: {ratio:.3f}x")
    
    if ratio > 1:
        print(f"  Need to increase weights/input by {ratio:.2f}x to reach target")
    else:
        print(f"  Need to decrease weights/input by {1/ratio:.2f}x to reach target")
    
    return ratio

def create_neurons(neuron_configs, simulation_params, connections=None):
    np.random.seed(2025)
    duration = simulation_params.get('duration', 1.0) 
    dt = float(simulation_params.get('dt', 1))  

    try:
        neuron_groups = {}

        for config in neuron_configs:
            name = config['name']
            N = config['N']

            if config.get('neuron_type', None) == 'poisson':
                if 'target_rates' in config and len(config['target_rates']) > 0:
                    target, rate_info = list(config['target_rates'].items())[0]
                    rate_expr = rate_info['equation']
                    N = config['N']
                    total_rate = float(eval(rate_expr, {"Hz": Hz}))
                    rate_per_neuron = total_rate / N
                    group = PoissonGroup(N, rates=rate_per_neuron * Hz)
                    neuron_groups[name] = group
                continue

            if 'model_class' in config and 'params_file' in config:
                params = load_params_from_file(config['params_file'])
                module_name = f"Neuronmodels.{config['model_class']}"
                model_module = importlib.import_module(module_name)
                model_class = getattr(model_module, config['model_class'])
                model_instance = model_class(N, params, connections)
                neuron_group = model_instance.create_neurons()
                neuron_groups[name] = neuron_group

        return neuron_groups

    except Exception as e:
        print(f"Error creating neuron groups: {str(e)}")
        print(f"Failed configuration: {config}")
        raise