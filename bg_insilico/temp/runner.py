from brian2 import *
from module.models.neuron_models import create_neurons
from module.models.Synapse import create_synapses
from module.utils.data_handler import plot_raster, plot_membrane_potential
from brian2 import profiling_summary

import json 
import numpy as np

def run_simulation_with_inh_ext_input(neuron_configs, connections, synapse_class, simulation_params, plot_order=None):
    
    def get_neuron_params(configs, name):
        for config in configs:
            if config['name'] == name:
                with open(config['params_file'], 'r') as f:
                    params = json.load(f)
                return params
        return None
    
    try:
        start_scope()
        
        neuron_groups = create_neurons(neuron_configs)
        print(f"Neuron Groups: {neuron_groups.keys()}") 
        synapse_connections = create_synapses(neuron_groups, connections, synapse_class)

        net = Network()
        net.add(neuron_groups.values())
        net.add(synapse_connections)
        
        spike_monitors = {}
        voltage_monitors = {}
        
        for name, group in neuron_groups.items():
            spike_mon = SpikeMonitor(group)
            spike_monitors[name] = spike_mon
            net.add(spike_mon)
            
            if 'v' in group.variables:
                voltage_mon = StateMonitor(group, 'v', record=True)
                voltage_monitors[name] = voltage_mon
                net.add(voltage_mon)

        duration = simulation_params['duration'] * ms
        
        """
        for t in range(0, int(duration/ms), 100):  
            print(f"Remaining time: {duration/ms - t} ms")
            net.run(100 * ms, profile=True)    
        """
        net.run(duration)

        for name, monitor in voltage_monitors.items():
            if monitor.v.size > 0:
                print(f"{name} - Min Voltage: {np.min(monitor.v) / mV} mV")
                print(f"{name} - Max Voltage: {np.max(monitor.v) / mV} mV")
            else:
                print(f"Warning: No voltage data recorded for {name}")

        if plot_order:
            plot_raster(spike_monitors, plot_order)
        else:
            plot_raster(spike_monitors)
        
        plot_membrane_potential(voltage_monitors)

        results = {
            'spike_monitors': spike_monitors,
            'voltage_monitors': voltage_monitors,
            'neuron_groups': neuron_groups,
            'synapse_connections': synapse_connections
        }
        return results
        
    except Exception as e:
        print(f"Error during simulation: {str(e)}")
        raise