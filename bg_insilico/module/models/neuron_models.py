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

def distribute_sequential_firing_rates(N, total_rate):
    total_events = int(total_rate)  
    event_indices = np.arange(total_events)
    neuron_events = np.array_split(event_indices, N)  
    neuron_rates = [len(events) for events in neuron_events]  
    return neuron_rates * Hz


def generate_non_overlapping_poisson_input(N, total_rate, duration, dt=1 * ms):

    try:
        total_spikes = int(total_rate * duration) 
        spike_times = np.sort(np.random.uniform(0, duration, total_spikes))  
        spike_times = np.round(spike_times / dt) * dt  
        neuron_indices = np.random.choice(N, total_spikes, replace=True)

        unique_spikes = set(zip(neuron_indices, spike_times))
        neuron_indices, spike_times = zip(*unique_spikes)

        return np.array(neuron_indices), np.array(spike_times)

    except Exception as e:
        print(f"Error in generate_non_overlapping_poisson_input: {str(e)}")
        raise

def create_neurons(neuron_configs, simulation_params, connections=None):
    try:
        neuron_groups = {}
        
        for config in neuron_configs:
            name = config['name']
            N = config['N']

            if name == 'Cortex':
                if 'target_rates' in config:
                    for target, rate_info in config['target_rates'].items():
                        target_N = get_neuron_count(neuron_configs, target)

                        if target_N is None or target_N == 0:
                            print(f"Warning: Could not find valid neuron count for target '{target}'")
                            continue

                        t = defaultclock.t 
                        total_rate = eval(rate_info['equation'], {"t": t, "Hz": Hz, "ms": ms})
                        rate_per_neuron = total_rate 
                        neuron_group = PoissonGroup(
                            target_N, 
                            rates=rate_per_neuron  
                        )

                        neuron_groups[f'Cortex_{target}'] = neuron_group
                        print(f"Created Cortex input for {target} with {rate_per_neuron} Hz per neuron")

                continue

            if name == 'Ext':
                if 'target_rates' in config:
                    for target, rate_info in config['target_rates'].items():
                        target_N = get_neuron_count(neuron_configs, target)

                        if target_N is None:
                            print(f"Warning: Could not find neuron count for target '{target}'")
                            continue

                        total_rate = eval(rate_info['equation']) 
                        rate_per_neuron = total_rate / target_N 
                        ext_poisson_group = PoissonGroup(target_N, rates=rate_per_neuron)

                        neuron_groups[f'Ext_{target}'] = ext_poisson_group

                        print(f"Created Ext Poisson input for {target} with {rate_per_neuron} Hz per neuron")

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
