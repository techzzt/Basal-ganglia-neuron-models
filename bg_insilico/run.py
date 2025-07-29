#!/usr/bin/env python3
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
from module.utils.visualization import (analyze_firing_rates_by_stimulus_periods, plot_improved_overall_raster,
                                       plot_continuous_firing_rate_with_samples, plot_multi_neuron_stimulus_overview,
                                       plot_firing_rate_fft_multi_page, plot_membrane_zoom, plot_raster_zoom,
                                       analyze_input_rates_and_spike_counts, plot_multi_neuron_membrane_potential_comparison)

def parse_arguments():
    parser = argparse.ArgumentParser(
        description='Run basal ganglia simulation with specified parameters',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
        Examples:
        python run.py                                  
        python run.py --config config/test_normal_noin.json
        python run.py -c config/test_dop_noin.json
        python run.py --list-configs                   
        python run.py -l                              
        """
    )
    parser.add_argument('--config', '-c', 
                       default='config/test_dop_noin.json',
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
    
    # Check if config file exists
    if not os.path.exists(params_file):
        print(f"Error: Configuration file '{params_file}' not found!")
        print("Available options:")
        list_available_configs()
        return
    
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
    
    zoom_windows = params.get('zoom_windows', {})
    first_window = tuple(zoom_windows.get('first_window', [3800, 4000])) * ms
    last_window = tuple(zoom_windows.get('last_window', [8500, 8700])) * ms

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
    

    
    stimulus_enabled = simulation_params.get('stimulus', {}).get('enabled', False)
    if stimulus_enabled:
        stimulus_config = simulation_params['stimulus']
        stim_start_time = stimulus_config.get('start_time', 0)
        stim_duration = stimulus_config.get('duration', 0)
        stim_end_time = stim_start_time + stim_duration
        print(f"Stimulus: {stim_start_time}-{stim_end_time}ms")
        
        analyze_firing_rates_by_stimulus_periods(
            results['spike_monitors'],
            stimulus_config,
            analysis_start_time,
            plot_order,
            params.get('display_names', None)
        )
    else:
        print("Stimulus: disabled")
    
    # 입력 rate와 spike count 분석
    analyze_input_rates_and_spike_counts(
        results['spike_monitors'],
        ext_inputs,
        neuron_configs,
        simulation_params.get('stimulus', {}),
        analysis_start_time,
        analysis_end_time
    )

        
    stimulus_periods = []
    if 'external_inputs' in simulation_params and 'poisson_trains' in simulation_params['external_inputs']:
        for train_config in simulation_params['external_inputs']['poisson_trains']:
            if 'active_periods' in train_config:
                for period in train_config['active_periods']:
                    start_time = period['start'] * ms
                    end_time = period['end'] * ms
                    stimulus_periods.append((start_time, end_time))

    plot_continuous_firing_rate_with_samples(results['spike_monitors'], start_time=analysis_start_time, end_time=analysis_end_time, bin_size=10*ms, 
                                            plot_order=plot_order, display_names=params.get('display_names', None), stimulus_config=stimulus_config, 
                                            smooth_sigma=3, save_plot=False, n_samples=10, neurons_per_sample=30)
    
    plot_improved_overall_raster(
        results['spike_monitors'],
        sample_size=10, 
        plot_order=plot_order,
        start_time=analysis_start_time,
        end_time=analysis_end_time,
        display_names=params.get('display_names', None),
        stimulus_periods=stimulus_periods,
        save_plot=True
    )
        
    thresholds = {
        'FSN': 25.0,
        'MSND1': 40,
        'MSND2': 40,
        'GPeT1': 15,
        'GPeTA': 15,
        'STN': 15,
        'SNr': 20,
    }
    

    plot_multi_neuron_stimulus_overview(
        results['voltage_monitors'],
        results['spike_monitors'],
        simulation_params.get('stimulus', {}),
        target_groups=['FSN', 'MSND1', 'MSND2', 'GPeT1', 'GPeTA', 'STN', 'SNr'],
        neurons_per_group=1, 
        analysis_window=(analysis_start_time, analysis_end_time), 
        unified_y_scale=True,  
        threshold_clipping=True, 
        display_names=params.get('display_names', None),
        thresholds=thresholds
    )
    
    # 여러 뉴런의 membrane potential 비교 분석
    plot_multi_neuron_membrane_potential_comparison(
        results['voltage_monitors'],
        results['spike_monitors'],
        target_groups=['FSN', 'MSND1', 'MSND2', 'GPeT1', 'GPeTA', 'STN', 'SNr'],
        neurons_per_group=5,
        analysis_window=(analysis_start_time, analysis_end_time),
        display_names=params.get('display_names', None),
        thresholds=thresholds
    )

    plot_firing_rate_fft_multi_page(
        results['spike_monitors'],
        neuron_indices=None,
        start_time=0*ms,
        end_time=10000*ms,
        bin_size=10*ms,
        show_mean=True,
        max_freq=100,
        title='Firing Rate FFT Spectra',
        display_names=params.get('display_names', None)
    )

    for group_name, vmon in results['voltage_monitors'].items():
        plot_membrane_zoom(vmon, time_window=first_window, neuron_indices=[0], group_name=group_name, 
                          spike_monitors=results['spike_monitors'], thresholds=thresholds,
                          display_names=params.get('display_names', None))  
        plot_membrane_zoom(vmon, time_window=last_window, neuron_indices=[0], group_name=group_name,
                          spike_monitors=results['spike_monitors'], thresholds=thresholds,
                          display_names=params.get('display_names', None))

    for group_name, smon in results['spike_monitors'].items():
        plot_raster_zoom(smon, time_window=first_window, neuron_indices=None, group_name=group_name,
                        display_names=params.get('display_names', None)) 
        plot_raster_zoom(smon, time_window=last_window, neuron_indices=None, group_name=group_name,
                        display_names=params.get('display_names', None))  
    


if __name__ == "__main__":
    main()