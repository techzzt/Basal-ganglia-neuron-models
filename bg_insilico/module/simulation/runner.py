from brian2 import *
# set_device('cpp_standalone')

from module.models.neuron_models import create_neurons
from module.models.Synapse import create_synapses
from module.utils.data_handler import plot_raster, plot_membrane_potential, plot_single_neuron_raster, plot_raster_all_neurons_stim_window, plot_isyn, plot_conductance

import json 
import numpy as np

# https://brian.discourse.group/t/cannot-use-cython/895/4

import os
os.environ['CC'] = 'gcc'
os.environ['CXX'] = 'g++'

def run_simulation_with_inh_ext_input(neuron_configs, connections, synapse_class, simulation_params, plot_order=None):
    
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
        # conductance_monitors = {}

        for name, group in neuron_groups.items():
            spike_mon = SpikeMonitor(group)
            spike_monitors[name] = spike_mon
            net.add(spike_mon)
            
            if 'v' in group.variables and 'Isyn' in group.variables:
                voltage_mon = StateMonitor(group, ['v', 'Isyn'], record=[0])
            elif 'v' in group.variables:
                voltage_mon = StateMonitor(group, 'v', record=[0])
            elif 'Isyn' in group.variables:
                voltage_mon = StateMonitor(group, 'Isyn', record=[0])
            else:
                continue

            voltage_monitors[name] = voltage_mon
            net.add(voltage_mon)
        
        """
            if 'g_a' in group.variables:
                g_mon = StateMonitor(group, ['g_a'], record=[0])
                conductance_monitors[f'{name}_g_a'] = g_mon
                net.add(g_mon)

            if 'g_n' in group.variables:
                g_mon = StateMonitor(group, ['g_n'], record=[0])
                conductance_monitors[f'{name}_g_n'] = g_mon
                net.add(g_mon)

            if 'g_g' in group.variables:
                g_mon = StateMonitor(group, ['g_g'], record=[0])
                conductance_monitors[f'{name}_g_g'] = g_mon
                net.add(g_mon)
        """        
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

        plot_raster(spike_monitors, 30, plot_order) 
        plot_membrane_potential(voltage_monitors, plot_order)
        # plot_single_neuron_raster(spike_monitors, 10, plot_order)
        plot_raster_all_neurons_stim_window(spike_monitors, 1000*ms, 10000 * ms, plot_order)
        plot_isyn(voltage_monitors, plot_order)
        # plot_conductance(results['conductance_monitors'], name='MSND1')

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