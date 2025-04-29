#!/usr/bin/env python3
import json
from module.simulation.runner import run_simulation_with_inh_ext_input
from module.utils.param_loader import load_params
from brian2 import ms 

def main():
    params_file = 'config/test_normal_msn_params_noin.json'
    
    params = load_params(params_file)
    neuron_configs = params['neurons']
    connections = params['connections']   
    synapse_class = params['synapse_class']
    simulation_params = params['simulation']
    plot_order = params['plot_order']
    start_time = params.get('start_time', 0) * ms
    end_time = params.get('end_time', 10000) * ms

    # Simulation 
    results = run_simulation_with_inh_ext_input(
        neuron_configs=neuron_configs,
        connections=connections,
        synapse_class=synapse_class,
        simulation_params=simulation_params,    
        plot_order=plot_order, 
        start_time=start_time,
        end_time=end_time
        )
    
    print("Simulation completed successfully")

if __name__ == "__main__":
    main()