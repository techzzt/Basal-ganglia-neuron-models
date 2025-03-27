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

def generate_non_overlapping_poisson_input(N, total_rate, duration):

    total_spikes = int(total_rate * duration)  
    spike_times = np.sort(np.random.uniform(0, duration, total_spikes))  

    neuron_spikes = [[] for _ in range(N)] 
    assigned_spikes = set() 

    for i in range(N):
        available_spikes = [t for t in spike_times if t not in assigned_spikes] 
        if len(available_spikes) == 0:
            break 
        
        num_spikes = min(len(available_spikes), len(available_spikes) // (N - i)) 
    
        selected_spikes = np.random.choice(available_spikes, num_spikes, replace=False)
        neuron_spikes[i] = sorted(selected_spikes)
        assigned_spikes.update(selected_spikes)

    return neuron_spikes


def create_neurons(neuron_configs, connections=None):
    try:
        neuron_groups = {}

        for config in neuron_configs:
            name = config['name']
            N = config['N']

            if name == 'Cortex':
                if 'target_rates' in config:
                    for target, rate_info in config['target_rates'].items():
                        target_N = get_neuron_count(neuron_configs, target)

                        if target_N is None:
                            print(f"Warning: Could not find neuron count for target '{target}'")
                            continue

                        total_rate = eval(rate_info['equation'])
                        print(f"Total rate for {target}: {total_rate} Hz")

                        spike_times_per_neuron = generate_non_overlapping_poisson_input(target_N, total_rate, duration=1000*ms)

                        neuron_indices = []
                        spike_times = []

                        for neuron_idx, spikes in enumerate(spike_times_per_neuron):
                            unique_spikes = np.unique(spikes)  
                            neuron_indices.extend([neuron_idx] * len(unique_spikes))
                            spike_times.extend(unique_spikes)


                        cortex_spike_gen = SpikeGeneratorGroup(target_N, neuron_indices, spike_times * second)

                        neuron_groups[f'Cortex_{target}'] = cortex_spike_gen

                        print(f"Generated non-overlapping Poisson input for Cortex_{target} with total rate {total_rate} Hz")

                continue

            if name == 'Ext':
                if 'target_rates' in config:
                    for target, rate_info in config['target_rates'].items():
                        target_N = get_neuron_count(neuron_configs, target)

                        if target_N is None:
                            print(f"Warning: Could not find neuron count for target '{target}'")
                            continue

                        total_rate = eval(rate_info['equation']) 
                        print(f"Total rate for {target}: {total_rate} Hz (Ext Poisson input)")

                        ext_poisson_group = PoissonGroup(target_N, rates=total_rate)

                        neuron_groups[f'Ext_{target}'] = ext_poisson_group

                        print(f"Created Ext Poisson input for {target} with rate {total_rate}")

                continue

            if 'model_class' in config and 'params_file' in config:
                params = load_params_from_file(config['params_file'])

                module_name = f"Neuronmodels.{config['model_class']}"
                model_module = importlib.import_module(module_name)
                model_class = getattr(model_module, config['model_class'])

                model_instance = model_class(N, params, connections)
                neuron_group = model_instance.create_neurons()

                neuron_groups[name] = neuron_group
                print(f"Created {name} neurons using {config['model_class']}")

        return neuron_groups

    except Exception as e:
        print(f"Error creating neuron groups: {str(e)}")
        print(f"Failed configuration: {config}")
        raise
