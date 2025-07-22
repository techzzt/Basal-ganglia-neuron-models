#!/usr/bin/env python3
import json
import numpy as np
import sys
import os
from brian2 import ms

from module.simulation.runner import run_simulation_with_inh_ext_input
from module.utils.param_loader import load_params
from module.utils.visualization import (analyze_firing_rates_by_stimulus_periods, plot_continuous_firing_rate, plot_improved_overall_raster,
                                       plot_continuous_firing_rate_with_samples, plot_enhanced_multi_neuron_stimulus_overview, 
                                       plot_place_cell_theta_analysis)
def main():
    params_file = 'config/test_normal_noin.json'
    
    save_isi_ranges = True    
    use_saved_ranges = False 
    ranges_filename = 'isi_axis_ranges.json'  
    
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
    
    print("Simulation completed successfully")
    
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
        
        plot_continuous_firing_rate(
            results['spike_monitors'],
            start_time=analysis_start_time,
            end_time=analysis_end_time,  
            bin_size=10*ms,  
            plot_order=plot_order,
            display_names=params.get('display_names', None),
            stimulus_config=stimulus_config,
            smooth_sigma=3,  
            show_confidence=True,  
            layout_mode='multi', 
            plots_per_page=6
        )
        
        plot_place_cell_theta_analysis(
            results['spike_monitors'],
            start_time=analysis_start_time,
            end_time=min(analysis_end_time, analysis_start_time + 5000*ms), 
            plot_order=plot_order,
            display_names=params.get('display_names', None),
            stimulus_config=stimulus_config,
            save_plot=True,
            place_field_center=0.0, 
            spatial_range=3.0  
        )
    else:
        print("Stimulus: disabled")

        
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
    
    plot_enhanced_multi_neuron_stimulus_overview(
        results['voltage_monitors'],
        results['spike_monitors'],
        simulation_params.get('stimulus', {}),
        target_groups=['FSN', 'MSND1', 'MSND2', 'GPeT1', 'GPeTA', 'STN', 'SNr'],
        neurons_per_group=2,
        analysis_window=(analysis_start_time, min(analysis_end_time, analysis_start_time + 5000*ms)),
        unified_y_scale=True,
        threshold_clipping=True,
        display_names=params.get('display_names', None),
        thresholds=thresholds
    )

    try:
        pass
    
    except Exception as e:
        print(f"Error during spike interval analysis: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()