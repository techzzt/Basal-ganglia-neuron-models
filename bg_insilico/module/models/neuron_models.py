from brian2 import *
import importlib
import json
import numpy as np 
from brian2 import mV, ms, nS, Hz

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

    # time-dependent total input rate 
    total_rate_values = np.array([
        eval(rate_expr, {"t": t, "Hz": Hz, "ms": ms, "second": second, "np": np})
        for t in time_points
    ]) 

    per_neuron_rate_values = total_rate_values / N
    rate_array = TimedArray(per_neuron_rate_values * Hz, dt)

    return PoissonGroup(N, rates='rate_array(t)', namespace={'rate_array': rate_array})


def generate_non_overlapping_poisson_input(N, total_rate, duration, dt=1 * ms, chunk_size=1000):

    try:
        duration = duration * second 
        total_chunks = int(np.ceil(duration / (chunk_size * dt)))
        all_indices = []
        all_times = []
        
        for chunk in range(total_chunks):
            start_time = float(chunk * chunk_size * dt / second)
            end_time = float(min((chunk + 1) * chunk_size * dt, duration) / second)
            chunk_duration = end_time - start_time
            
            chunk_spikes = int(total_rate * chunk_duration)
            if chunk_spikes == 0:
                continue
                
            times = np.sort(np.random.uniform(start_time, end_time, chunk_spikes))
            times = np.round(times / float(dt/second)) * float(dt/second)
            indices = np.random.choice(N, chunk_spikes, replace=True)
            
            unique_spikes = set(zip(indices, times))
            if unique_spikes:
                chunk_indices, chunk_times = zip(*unique_spikes)
                all_indices.extend(chunk_indices)
                all_times.extend(chunk_times)
        
        return np.array(all_indices), np.array(all_times)
    
    except Exception as e:
        print(f"Error in generate_non_overlapping_poisson_input: {str(e)}")
        raise

def generate_non_overlapping_poisson_input_timevarying(N, rate_expr, duration, dt=1*ms, chunk_size=1000):

    try:
        duration = float(duration) * second  
        all_indices = []
        all_times = []
        total_chunks = int(np.ceil(duration / (chunk_size * dt)))
        
        for chunk in range(total_chunks):
            start_time = chunk * chunk_size * dt
            end_time = min((chunk + 1) * chunk_size * dt, duration)
            chunk_times = np.arange(float(start_time/second), float(end_time/second), float(dt/second)) * second
            
            rates = np.array([
                eval(rate_expr, {"t": t, "Hz": Hz, "ms": ms, "second": second, "np": np})
                for t in chunk_times
            ])
            mean_rate = np.mean(rates)
            
            n_spikes = np.random.poisson(float(mean_rate * len(chunk_times) * dt))
            if n_spikes > 0:
                times = np.random.choice(chunk_times, n_spikes, p=rates/np.sum(rates))
                indices = np.random.choice(N, n_spikes, replace=True)
                
                unique_spikes = set(zip(indices, [float(t/second) for t in times]))
                if unique_spikes:
                    chunk_indices, chunk_times = zip(*unique_spikes)
                    all_indices.extend(chunk_indices)
                    all_times.extend(chunk_times)
        
        return np.array(all_indices), np.array(all_times)
    
    except Exception as e:
        print(f"Error in generate_non_overlapping_poisson_input_timevarying: {str(e)}")
        raise

def create_neurons(neuron_configs, simulation_params, connections=None):
    np.random.seed(2025)
    duration = simulation_params.get('duration', 1.0) 
    dt = float(simulation_params.get('dt', 0.001))  

    try:
        neuron_groups = {}

        for config in neuron_configs:
            name = config['name']
            N = config['N']

            if name in ['Cortex', 'Ext']:
                if 'target_rates' in config:
                    for target, rate_info in config['target_rates'].items():
                        target_N = get_neuron_count(neuron_configs, target)
                        if target_N is None or target_N == 0:
                            print(f"Warning: Could not find valid neuron count for target '{target}'")
                            continue

                        rate_expr = rate_info['equation']
                        if 't' in rate_expr:
                            # Time-varying rate
                            total_rate = eval(rate_expr, {"Hz": Hz, "ms": ms, "second": second, "np": np})
                            population_rate = float(total_rate/Hz) * target_N  # Total population rate
                            group = PoissonGroup(N, rates=f'rand()*{2*population_rate/N}*Hz')  # Random variation around mean
                        else:
                            # Constant rate
                            total_rate = float(eval(rate_expr, {"Hz": Hz}))
                            population_rate = total_rate * target_N  
                            group = PoissonGroup(N, rates=f'rand()*{2*population_rate/N}*Hz')  # Random variation around mean
                            
                            print(f"Target rate for {name}->{target}:")
                            print(f"  - Target rate per neuron: {total_rate} Hz")
                            print(f"  - Average rate per input: {population_rate/N:.3f} Hz")
                        
                        neuron_groups[f'{name}_{target}'] = group
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
