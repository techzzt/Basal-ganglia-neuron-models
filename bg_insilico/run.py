#!/usr/bin/env python3
import json
from module.simulation.runner import run_simulation_with_inh_ext_input
from module.utils.param_loader import load_params

def main():
    # parameter set 
    params_file = 'config/params.json'
    
    params = load_params(params_file)
    neuron_configs = params['neurons']
    synapse_params = params['synapse_params']
    cortex_inputs = params['cortex_inputs']
    synapse_class = params['synapse_class']

    # Simulation 
    results = run_simulation_with_inh_ext_input(
        neuron_configs=neuron_configs,
        synapse_params=synapse_params,
        synapse_class=synapse_class,
        cortex_inputs=cortex_inputs
    )
    
    print("Simulation completed successfully")

if __name__ == "__main__":
    main()