#!/usr/bin/env python3
# Copyright (c) 2025. All rights reserved.
# Author: keun (Jieun Kim)

import json
import numpy as np
import sys
import os
import argparse
from brian2 import ms
import gc
gc.collect()

from module.simulation.runner import run_simulation_with_inh_ext_input
from module.utils.param_loader import load_params
from module.utils.visualization import plot_improved_overall_raster, plot_firing_rate_fft_multi_page

def parse_arguments():
    parser = argparse.ArgumentParser(
        description='Run simulation',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
        Examples:
        python run.py                                  
        python run.py --config config/test_dop_noin.json                           
        """
    )
    parser.add_argument('--config', '-c', 
                       default='config/test_normal_noin.json',
                       help='Path to configuration JSON file (default: config/test_dop_noin.json)')
    parser.add_argument('--list-configs', '-l', 
                       action='store_true',
                       help='List available configuration files')
    return parser.parse_args()

def list_available_configs():
    config_dir = 'config'
    if os.path.exists(config_dir):
        config_files = [f for f in os.listdir(config_dir) if f.endswith('.json')]
        print("Available configuration files:")
        for i, config in enumerate(sorted(config_files), 1):
            print(f"  {i}. {config}")
        print(f"\nUse: python run.py --config config/{config_files[0] if config_files else 'your_config.json'}")
    else:
        print(f"Config directory '{config_dir}' not found")

def main():
    args = parse_arguments()
    
    if args.list_configs:
        list_available_configs()
        return
    
    params_file = args.config
    
    if not os.path.exists(params_file):
        print(f"Error: Configuration file '{params_file}' not found!")
        list_available_configs()
        return
    
    # Load parameters
    params = load_params(params_file)
    neuron_configs = params['neurons']
    connections = params['connections']   
    synapse_class = params['synapse_class']
    simulation_params = params['simulation']
    plot_order = params['plot_order']
    analysis_start_time = params.get('start_time', 0) * ms
    analysis_end_time = params.get('end_time', 10000) * ms

    if 'display_names' in params:
        simulation_params['display_names'] = params['display_names']

    # Setup external inputs
    ext_inputs = {}
    for neuron_config in neuron_configs:
        if neuron_config.get('neuron_type') == 'poisson':
            name = neuron_config['name']
            if 'target_rates' in neuron_config:
                target, rate_info = list(neuron_config['target_rates'].items())[0]
                rate_expr = rate_info['equation']
                ext_inputs[target] = rate_expr 

    amplitude_oscillations = {
        'MSND1': 0.11,
        'MSND2': 0.11, 
        'FSN': 0.11,
        'STN': 0.11 
    }

    # Run simulation
    results = run_simulation_with_inh_ext_input(
        neuron_configs=neuron_configs,
        connections=connections,
        synapse_class=synapse_class,
        simulation_params=simulation_params,    
        plot_order=plot_order, 
        start_time=analysis_start_time,
        end_time=analysis_end_time,
        ext_inputs=ext_inputs,
        amplitude_oscillations=amplitude_oscillations  
    )

    # Plot results
    try:
        plot_improved_overall_raster(
            results['spike_monitors'],
            sample_size=15, 
            plot_order=plot_order,
            start_time=3000*ms,
            end_time=7000*ms,
            display_names=params.get('display_names', None),
            save_plot=True
        )
    except Exception as e:
        print(f"Error in raster plot: {e}")
        
    try:
        plot_firing_rate_fft_multi_page(
            results['spike_monitors'],
            neuron_indices=None,
            start_time=0*ms,
            end_time=10000*ms,
            bin_size=10*ms,
            show_mean=True,
            max_freq=60,
            title='Firing Rate FFT Spectra',
            display_names=params.get('display_names', None)
        )
    except Exception as e:
        print(f"Error in FFT plot: {e}")

    # Save results
    config_file = args.config
    if 'normal' in config_file.lower():
        try:
            import pickle
            save_data = {
                'meta': {
                    'config': os.path.basename(config_file),
                    'start_time_ms': int(analysis_start_time/ms),
                    'end_time_ms': int(analysis_end_time/ms),
                    'default_bin_size_ms': 10
                },
                'groups': []
            }

            for group_name, monitor in results['spike_monitors'].items():
                N_group = int(monitor.source.N)
                t_ms_arr = np.array(monitor.t / ms, dtype=float)
                i_arr = np.array(monitor.i, dtype=int)
                save_data[f'spike_monitors_{group_name}'] = {
                    't_ms': t_ms_arr,
                    'i': i_arr,
                    'N': N_group
                }
                save_data['groups'].append(group_name)

            with open('normal_results.pkl', 'wb') as f:
                pickle.dump(save_data, f)
            print("Results saved to 'normal_results.pkl'")
        
        except Exception as e:
            print(f"Error saving normal results: {e}")
    
    elif 'dop' in config_file.lower():
        try:
            import pickle
            save_data = {
                'meta': {
                    'config': os.path.basename(config_file),
                    'start_time_ms': int(analysis_start_time/ms),
                    'end_time_ms': int(analysis_end_time/ms),
                    'default_bin_size_ms': 10
                },
                'groups': []
            }

            for group_name, monitor in results['spike_monitors'].items():
                N_group = int(monitor.source.N)
                t_ms_arr = np.array(monitor.t / ms, dtype=float)
                i_arr = np.array(monitor.i, dtype=int)
                save_data[f'spike_monitors_{group_name}'] = {
                    't_ms': t_ms_arr,
                    'i': i_arr,
                    'N': N_group
                }
                save_data['groups'].append(group_name)

            with open('pd_results.pkl', 'wb') as f:
                pickle.dump(save_data, f)
            print("Results saved to 'pd_results.pkl'")
        
        except Exception as e:
            print(f"Results Saving Error: {e}")

if __name__ == "__main__":
    main()