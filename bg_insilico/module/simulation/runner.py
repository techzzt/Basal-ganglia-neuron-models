from brian2 import *
from module.models.neuron_models import create_neurons
from module.models.Synapse import create_synapses
from module.utils.data_handler import (
    plot_raster, plot_membrane_potential,
    compute_firing_rates_all_neurons, 
    plot_raster_all_neurons_stim_window
)

from module.utils.sta import compute_sta 
import numpy as np
import os

def run_simulation_with_inh_ext_input(neuron_configs, connections, synapse_class, simulation_params, plot_order=None, start_time=0*ms, end_time=30000*ms):
    try:
        start_scope()

        neuron_groups = create_neurons(neuron_configs, simulation_params, connections)
        synapse_connections = create_synapses(neuron_groups, connections, synapse_class)

        net = Network()
        net.add(neuron_groups.values())
        net.add(synapse_connections)

        spike_monitors = {}
        voltage_monitors = {}

        record_neurons = plot_order if plot_order else neuron_groups.keys()

        for name, group in neuron_groups.items():
            if name in record_neurons:
                sp_mon = SpikeMonitor(group, record=[0])
                spike_monitors[name] = sp_mon
                net.add(sp_mon)

                if 'v' in group.variables:
                    v_mon = StateMonitor(group, 'v', record=[0], dt=10*ms)  # 기록 간격도 줄임
                    voltage_monitors[name] = v_mon
                    net.add(v_mon)

        # === Chunked Simulation ===
        duration = simulation_params['duration'] * ms
        chunk_size = 5000 * ms
        t = 0 * ms
        while t < duration:
            run_time = min(chunk_size, duration - t)
            print(f"Running simulation chunk: {t} to {t + run_time}")
            net.run(run_time)
            t += run_time

        # === Post-Simulation Analysis ===
        for name, monitor in voltage_monitors.items():
            if monitor.v.size > 0:
                v_vals = monitor.v[0] / mV
                # print(f"{name} - Min Voltage: {np.min(v_vals):.2f} mV, Max Voltage: {np.max(v_vals):.2f} mV")

        compute_firing_rates_all_neurons(spike_monitors, start_time=2000*ms, end_time=end_time, plot_order=plot_order)
        plot_raster(spike_monitors, sample_size=30, plot_order=plot_order, start_time=start_time, end_time=end_time)
        # plot_membrane_potential(voltage_monitors, plot_order)
        # plot_raster_all_neurons_stim_window(spike_monitors, stim_start=2000*ms, end_time=end_time, plot_order=plot_order)
        
        """
        sta_results, bins = compute_sta(
            pre_monitors={k: spike_monitors[k] for k in relevant_pres if k in spike_monitors},
            post_monitors={k: spike_monitors[k] for k in plot_order if k in spike_monitors},
            neuron_groups=neuron_groups,
            synapses=synapse_connections, 
            connections=connections,
            start_from_end=5000*ms,
            window=30*ms
        )
        """

        return {
            'spike_monitors': spike_monitors,
            'voltage_monitors': voltage_monitors,
            'neuron_groups': neuron_groups,
            'synapse_connections': synapse_connections
        }

    except Exception as e:
        print(f"Error during simulation: {str(e)}")
        raise
