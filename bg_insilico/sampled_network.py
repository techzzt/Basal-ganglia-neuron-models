#!/usr/bin/env python3
import json
import time
from module.simulation.runner import run_simulation_with_inh_ext_input
from module.utils.param_loader import load_params
from brian2 import ms 

def main():    
     
    params = load_params('config/test_normal_noin.json')
    params['simulation']['duration'] = 10000 
    
    for neuron in params['neurons']:
        if 'N' in neuron:
            original_n = neuron['N']
            sampled_n = max(5, int(original_n / 5))
            neuron['N'] = sampled_n
    
    print(f'\nSimulation time: {params["simulation"]["duration"]}ms')
    
    ext_inputs = {}
    for neuron_config in params['neurons']:
        if neuron_config.get('neuron_type') == 'poisson':
            if 'target_rates' in neuron_config:
                target, rate_info = list(neuron_config['target_rates'].items())[0]
                rate_expr = rate_info['equation']
                ext_inputs[target] = {'rate': rate_expr}
                print(f'External input: {target} <- {rate_expr}')
    
    results = run_simulation_with_inh_ext_input(
        neuron_configs=params['neurons'],
        connections=params['connections'],
        synapse_class=params['synapse_class'],
        simulation_params=params['simulation'],
        plot_order=params['plot_order'],
        start_time=2000*ms, 
        end_time=params['simulation']['duration']*ms,  
        ext_inputs=ext_inputs
    )
    
    
    if 'firing_rates' in results:
        print(f'Work window: {2000}ms ~ {params["simulation"]["duration"]}ms')
        for neuron_type, rate in results['firing_rates'].items():
            print(f'   {neuron_type}: {rate:.2f} Hz')
    
    return results

if __name__ == "__main__":
    main() 