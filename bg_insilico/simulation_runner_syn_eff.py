from brian2 import *

from Neuronmodels import STN, GPeTA, GPeT1, FSN, MSND1, MSND2, SNr, Synapse

import matplotlib.pyplot as plt
import importlib
import numpy as np
import json
import plotly.graph_objects as go

def load_params(json_file):
    with open(json_file, 'r') as f:
        data = json.load(f)
    params = data['params']
    model_name = data['model']
    
    # Extract 'N' from params
    N = params.pop('N')['value'] 
    return N, params, model_name


def convert_units(params):
    converted_params = {}
    for param, info in params.items():
        value = info['value']
        unit = info['unit']
        if unit == 'nS':
            value *= nS
        elif unit == 'mV':
            value *= mV
        elif unit == 'ms':
            value *= ms
        elif unit == 'pF':
            value *= pF
        elif unit == 'pA':
            value *= pA
        elif unit == 'Hz':
            value *= Hz
        elif unit == 'Ohm':
            value *= ohm
        converted_params[param] = value
    return converted_params


def run_simulation_with_inh_ext_input(neuron_configs, synapse_params, synapse_class, simulation_duration=1000*ms):
    neurons = {}
    monitors = {}
    
    cortex_config = next((config for config in neuron_configs if config['name'] == 'Cortex'), None)

    for config in neuron_configs:
        if config.get('neuron_type') == 'poisson':
            # multiple cortex
            if isinstance(config.get('target_rates'), dict):
                neurons['Cortex'] = PoissonGroup(
                    config['N'],
                    rates=config['target_rates'].get('default', {}).get(
                        'equation',
                        '50*Hz + (t >= 200*ms) * (t < 400*ms) * 200*Hz + 3*Hz * randn()'
                    )
                )
                monitors['Cortex_spikes'] = SpikeMonitor(neurons['Cortex'])
                
                # cell - cortex
                for target, rate_info in config['target_rates'].items():
                    if target != 'default':
                        cortex_name = f"Cortex_{target}"
                        neurons[cortex_name] = PoissonGroup(
                            config['N'], 
                            rates=rate_info['equation']
                        )
                        monitors[f'{cortex_name}_spikes'] = SpikeMonitor(neurons[cortex_name])
            else:
                # single cortex
                neurons[config['name']] = PoissonGroup(
                    config['N'], 
                    rates=config.get('rate_equation', '50*Hz')
                )
                monitors[f'{config["name"]}_spikes'] = SpikeMonitor(neurons[config['name']])
            
        else:  # regular neurons
            try:
                _, params, model_name = load_params(config['params_file'])
                params_converted = convert_units(params)
                
                model_module = importlib.import_module(f'Neuronmodels.{config["model_class"]}')
                model = getattr(model_module, config['model_class'])(
                    N=config['N'], 
                    params=params_converted, 
                    neuron_type="E"
                )
                
                neurons[config['name']] = model.create_neurons()
                monitors[f'{config["name"]}_v'] = StateMonitor(neurons[config['name']], 'v', record=True)
                monitors[f'{config["name"]}_spikes'] = SpikeMonitor(neurons[config['name']])
            except KeyError as e:
                print(f"Error processing neuron config for {config['name']}: Missing {e}")
                raise

    synapse_module = importlib.import_module(f'Neuronmodels.{synapse_class}')
    synapse_instance = synapse_module.Synapse(neurons, synapse_params)
    synapse_connections = synapse_instance.create_synapse()

    # network
    net_components = (
        list(neurons.values()) + 
        list(synapse_connections) +  
        list(monitors.values())
    )    
    net = Network(*net_components)
    net.run(simulation_duration)

    # result 
    results = {
        'neurons': neurons,
        'monitors': monitors,
        'synapses': synapse_connections,
        'firing_rates': {}
    }
    
    # calculate firing rate 
    for name, spike_monitor in monitors.items():
        if 'spikes' in name:
            neuron_name = name.replace('_spikes', '')
            results['firing_rates'][neuron_name] = spike_monitor.count / (simulation_duration / second)

    return results

### Visualization spike 
def plot_raster(results, plot_order=None):
    spike_monitors = {
        name.replace('_spikes', ''): monitor 
        for name, monitor in results['monitors'].items() 
        if '_spikes' in name and not '_' in name.split('_spikes')[0]
    }
    
    if plot_order is None:
        plot_order = list(spike_monitors.keys())
        
    n_plots = len(plot_order)
    
    # Create figure
    fig, axes = plt.subplots(n_plots, 1, figsize=(12, 3*n_plots))
    if n_plots == 1:
        axes = [axes]
    
    # Plot each population
    for i, pop_name in enumerate(plot_order):
        if pop_name in spike_monitors:
            monitor = spike_monitors[pop_name]
            axes[i].scatter(monitor.t/ms, monitor.i, s=1)
            axes[i].set_title(f'{pop_name} Raster Plot')
            axes[i].set_ylabel('Neuron index')
            axes[i].set_xlim(0, 1000)
    
    axes[-1].set_xlabel('Time (ms)')
    plt.subplots_adjust(hspace=0.5)
    plt.show()

### Visualization membrane potential 
def plot_membrane_potentials(results, plot_order=None):
    if plot_order is None:
        plot_order = [name for name in results['neurons'].keys() if name != 'Cortex']
    
    plt.figure(figsize=(15, 3*len(plot_order)))
    colors = plt.cm.tab10(np.linspace(0, 1, len(plot_order)))
    
    for idx, neuron_name in enumerate(plot_order):
        plt.subplot(len(plot_order), 1, idx+1)
        v_monitor = results['monitors'][f'{neuron_name}_v']
        plt.plot(v_monitor.t/ms, v_monitor.v[0]/mV, 
                label=neuron_name, color=colors[idx])
        plt.title(f'{neuron_name} Membrane Potential')
        plt.ylabel('Membrane\nPotential (mV)')
        plt.grid(True)
        plt.legend()
    
    plt.xlabel('Time (ms)')
    plt.tight_layout()
    plt.show()