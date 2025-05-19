import numpy as np
import os
from copy import deepcopy

from brian2 import *
from module.models.neuron_models import create_neurons
from module.models.Synapse import create_synapses
from module.utils.visualization import (
    plot_raster, plot_membrane_potential,
    plot_raster_all_neurons_stim_window
    )

from module.utils.sta import compute_firing_rates_all_neurons, adjust_connection_weights, estimate_required_weight_adjustment

os.environ['CC'] = 'gcc'
os.environ['CXX'] = 'g++'
prefs.codegen.target = 'cython'

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
                sample_size = min(30, group.N)
                sample_indices = np.random.choice(group.N, size=sample_size, replace=False).tolist()
                sp_mon = SpikeMonitor(group, record=sample_indices)
                spike_monitors[name] = sp_mon
                net.add(sp_mon)

                if 'v' in group.variables:
                    v_mon = StateMonitor(group, 'v', record=sample_indices, dt=1*ms)  
                    voltage_monitors[name] = v_mon
                    net.add(v_mon)

        duration = simulation_params['duration'] * ms
        chunk_size = 1000 * ms
        t = 0 * ms
        while t < duration:
            run_time = min(chunk_size, duration - t)
            print(f"Running simulation chunk: {t} to {t + run_time}")
            net.run(run_time)
            t += run_time

        target_firing_rates = {
            'FSN': 15.0,
            'STN': 15.0,
            'GPeT1': 33.0,
            'GPeTA': 33.0,
            'MSND1': 0.1,
            'MSND2': 0.1
        }

        observed_rates = compute_firing_rates_all_neurons(
            spike_monitors,
            start_time=2000 * ms,
            end_time=end_time,
            plot_order=plot_order,
            return_dict=True
        )

        adjustment_factors = estimate_required_weight_adjustment(observed_rates, target_firing_rates)

        updated_connections = adjust_connection_weights(deepcopy(connections), adjustment_factors)
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
        print("\n=== Summary ===")
        print(f"{'Neuron':<10} | {'Observed (Hz)':<14} | {'Target (Hz)':<12} | {'Adj. Factor':<12}")
        for neuron in target_firing_rates:
            obs = observed_rates.get(neuron, 0.0)
            tgt = target_firing_rates[neuron]
            adj = adjustment_factors.get(neuron)
            obs_str = f"{obs:.2f}" if obs else "0.00"
            adj_str = f"{adj:.3f}" if adj else "N/A"
            print(f"{neuron:<10} | {obs_str:<14} | {tgt:<12} | {adj_str:<12}")

        plot_raster(spike_monitors, sample_size=30, plot_order=plot_order, start_time=start_time, end_time=end_time)

        return {
            'spike_monitors': spike_monitors,
            'voltage_monitors': voltage_monitors,
            'neuron_groups': neuron_groups,
            'synapse_connections': synapse_connections,
            'updated_connections': updated_connections,
            'observed_firing_rates': observed_rates,
            'adjustment_factors': adjustment_factors
        }

    except Exception as e:
        print(f"Error during simulation: {str(e)}")
        raise
