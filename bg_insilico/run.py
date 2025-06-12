#!/usr/bin/env python3
import json
from module.simulation.runner import run_simulation_with_inh_ext_input
from module.utils.param_loader import load_params
from brian2 import ms 

def main():
    params_file = 'config/test_normal_noin.json'
    
    params = load_params(params_file)
    neuron_configs = params['neurons']
    connections = params['connections']   
    synapse_class = params['synapse_class']
    simulation_params = params['simulation']
    plot_order = params['plot_order']
    start_time = params.get('start_time', 0) * ms
    end_time = params.get('end_time', 10000) * ms

    ext_inputs = {}
    for neuron_config in neuron_configs:
        if neuron_config.get('neuron_type') == 'poisson':
            name = neuron_config['name']
            if 'target_rates' in neuron_config:
                target, rate_info = list(neuron_config['target_rates'].items())[0]
                rate_expr = rate_info['equation']
                ext_inputs[target] = rate_expr  # Store rate_expr only
                # print(f"Setting up external input for {target}: {rate_expr}")

    # Default amplitude oscillation values
    amplitude_oscillations = {
        'MSND1': 0.11,
        'MSND2': 0.11, 
        'FSN': 0.11,
        'STN': 0.11  # slow-wave state
    }

    results = run_simulation_with_inh_ext_input(
        neuron_configs=neuron_configs,
        connections=connections,
        synapse_class=synapse_class,
        simulation_params=simulation_params,    
        plot_order=plot_order, 
        start_time=start_time,
        end_time=end_time,
        ext_inputs=ext_inputs,
        amplitude_oscillations=amplitude_oscillations  # Apply amplitude oscillation
        )
    
    print("Simulation completed successfully")

if __name__ == "__main__":
    main()