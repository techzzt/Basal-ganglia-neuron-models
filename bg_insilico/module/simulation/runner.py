from brian2 import *

from module.models.neuron_models import create_neurons
from module.models.Synapse import create_synapses
from module.utils.data_handler import plot_raster, plot_membrane_potential, compute_firing_rates_all_neurons, plot_raster_all_neurons_stim_window, plot_isyn
from module.utils.sta import compute_sta 

import json 
import numpy as np

# https://brian.discourse.group/t/cannot-use-cython/895/4

import os
os.environ['CC'] = 'gcc'
os.environ['CXX'] = 'g++'

def run_simulation_with_inh_ext_input(neuron_configs, connections, synapse_class, simulation_params, plot_order=None, start_time = 0 * ms, end_time = 10000 * ms):
    
    try:
        start_scope()
        
        neuron_groups = create_neurons(neuron_configs, simulation_params, connections)
        print(f"Neuron Groups: {neuron_groups.keys()}") 
        synapse_connections = create_synapses(neuron_groups, connections, synapse_class)

        net = Network()
        net.add(neuron_groups.values())
        net.add(synapse_connections)
        
        spike_monitors = {}
        voltage_monitors = {}
        start_time = start_time
        end_time = end_time

        for name, group in neuron_groups.items():
            spike_mon = SpikeMonitor(group)
            spike_monitors[name] = spike_mon
            net.add(spike_mon)
            
            if 'v' in group.variables and 'Isyn' in group.variables:
                voltage_mon = StateMonitor(group, ['v', 'Isyn'], record=[0], dt = 1*ms)
            elif 'v' in group.variables:
                voltage_mon = StateMonitor(group, 'v', record=[0], dt = 1*ms)
            elif 'Isyn' in group.variables:
                voltage_mon = StateMonitor(group, 'Isyn', record=[0], dt = 1*ms)
            else:
                continue

            voltage_monitors[name] = voltage_mon
            net.add(voltage_mon)
              
        duration = simulation_params['duration'] * ms

        net.run(duration)

        for name, monitor in voltage_monitors.items():

            if hasattr(monitor, 'v'):
                if monitor.v.size > 0:
                    print(f"{name} - Min Voltage: {np.min(monitor.v) / mV} mV")
                    print(f"{name} - Max Voltage: {np.max(monitor.v) / mV} mV")
                else:
                    print(f"Warning: No voltage data recorded for {name}")

            if hasattr(monitor, 'Isyn'):
                avg_current = np.mean(monitor.Isyn[0]) / pA
                print(f"{name} 평균 synaptic current: {avg_current:.2f} pA")

        compute_firing_rates_all_neurons(spike_monitors, start_time=2000*ms, end_time=end_time, plot_order=plot_order)
        plot_raster(spike_monitors, sample_size=30, plot_order=plot_order, start_time=start_time, end_time=end_time)
        plot_membrane_potential(voltage_monitors, plot_order)
        plot_raster_all_neurons_stim_window(spike_monitors, stim_start = 2000*ms, end_time=end_time, plot_order = plot_order)
        analysis_window = 100*ms  
        sta_start_time = duration - 5000*ms 

        post_monitors = {name: spike_monitors[name] for name in plot_order if name in spike_monitors}
        relevant_pres = {conn['pre'] for conn in connections.values() if conn['post'] in plot_order}
        pre_monitors = {name: spike_monitors[name] for name in relevant_pres if name in spike_monitors}

        sta_results, bins = compute_sta(
            pre_monitors=pre_monitors,
            post_monitors=post_monitors,
            neuron_groups=neuron_groups,
            synapses=synapse_connections, 
            connections=connections,
            start_from_end=5000*ms,
            window=30*ms
        )

        # plot_single_neuron_raster(spike_monitors, 10, plot_order)
        # plot_isyn(voltage_monitors, plot_order)

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