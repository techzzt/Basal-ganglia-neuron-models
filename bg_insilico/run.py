#!/usr/bin/env python3
import json
import numpy as np
import sys
import os
from brian2 import ms

from module.simulation.runner import run_simulation_with_inh_ext_input
from module.utils.param_loader import load_params
from module.utils.visualization import (analyze_firing_rates_by_stimulus_periods, plot_continuous_firing_rate, plot_improved_overall_raster,
                                       plot_circuit_flow_heatmap, plot_spike_burst_cascade, 
                                       plot_phase_space_only_overview, plot_enhanced_multi_neuron_stimulus_overview,
                                       plot_multi_sample_firing_rate_analysis, plot_place_cell_theta_analysis_custom,
                                       plot_individual_neuron_firing_rates, plot_place_cell_theta_analysis)
def main():
    params_file = 'config/test_dop_noin.json'
    
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
        
        plot_individual_neuron_firing_rates(
            results['spike_monitors'],
            start_time=analysis_start_time,
            end_time=min(analysis_end_time, analysis_start_time + 3000*ms),
            bin_size=20*ms,
            plot_order=plot_order,
            display_names=params.get('display_names', None),
            stimulus_config=stimulus_config,
            smooth_sigma=3,
            save_plot=True,
            neurons_per_group=10  
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
        
        plot_circuit_flow_heatmap(
            results['spike_monitors'],
            connections,
            start_time=analysis_start_time,
            end_time=min(analysis_end_time, analysis_start_time + 3000*ms), 
            bin_size=10*ms, 
            plot_order=plot_order,
            display_names=params.get('display_names', None),
            save_plot=True
        )

        plot_spike_burst_cascade(
            results['spike_monitors'],
            connections, 
            start_time=analysis_start_time,
            end_time=analysis_end_time, 
            burst_threshold=0.75,  
            cascade_window=50*ms, 
            plot_order=plot_order,
            display_names=params.get('display_names', None),
            save_plot=True
        )
        
    else:
        print("Stimulus: disabled")
        plot_circuit_flow_heatmap(
            results['spike_monitors'],
            connections,
            start_time=analysis_start_time,
            end_time=min(analysis_end_time, analysis_start_time + 3000*ms),  
            bin_size=10*ms, 
            plot_order=plot_order,
            display_names=params.get('display_names', None),
            save_plot=True
        )
        
        plot_spike_burst_cascade(
            results['spike_monitors'],
            connections, 
            start_time=analysis_start_time,
            end_time=analysis_end_time, 
            burst_threshold=0.75, 
            cascade_window=50*ms, 
            plot_order=plot_order,
            display_names=params.get('display_names', None),
            save_plot=True
        )
    
    if 'voltage_monitors' in results and results['voltage_monitors']:
        plot_phase_space_only_overview(
            results['voltage_monitors'],
            results['spike_monitors'],
            target_groups=['FSN', 'MSND1', 'MSND2', 'GPeT1', 'GPeTA', 'STN', 'GPi', 'SNr'],
            analysis_window=(analysis_start_time, min(analysis_end_time, analysis_start_time + 2000*ms)),
            display_names=params.get('display_names', None),
            save_plot=True
        )
    else:
        print("Voltage monitors not available - skipping single neuron analysis")
        
    stimulus_periods = []
    if 'external_inputs' in simulation_params and 'poisson_trains' in simulation_params['external_inputs']:
        for train_config in simulation_params['external_inputs']['poisson_trains']:
            if 'active_periods' in train_config:
                for period in train_config['active_periods']:
                    start_time = period['start'] * ms
                    end_time = period['end'] * ms
                    stimulus_periods.append((start_time, end_time))

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
    def generate_theta_phase(time_ms, frequency=8.0):
        theta_phase = 2 * np.pi * frequency * time_ms / 1000.0
        return theta_phase % (2 * np.pi)

    def calculate_spatial_position(time_ms, velocity=0.1):
        spatial_pos = velocity * (time_ms - time_ms[0])
        return spatial_pos

    def plot_place_cell_theta_analysis_all_cells(
        spike_monitor, 
        spatial_bins=40, phase_bins=40, spatial_range=3.0, 
        max_neurons=None 
    ):
        spike_times_all = spike_monitor.t / ms
        spike_indices_all = spike_monitor.i
        N = spike_monitor.source.N if max_neurons is None else min(max_neurons, spike_monitor.source.N)
        for neuron_idx in range(N):
            spike_times = spike_times_all[spike_indices_all == neuron_idx]
            if len(spike_times) == 0:
                continue
            spike_spatial_pos = calculate_spatial_position(spike_times)
            spike_theta_phase = generate_theta_phase(spike_times)
            plot_place_cell_theta_analysis_custom(
                spike_spatial_pos, spike_theta_phase, 
                spatial_bins=spatial_bins, phase_bins=phase_bins, spatial_range=spatial_range,
                save_path=f'place_cell_theta_neuron{neuron_idx}.png'
            )

    for group_name, monitor in results['spike_monitors'].items():
        plot_place_cell_theta_analysis_all_cells(monitor)
        
    plot_multi_sample_firing_rate_analysis(
        results['spike_monitors'],
        start_time=analysis_start_time,
        end_time=min(analysis_end_time, analysis_start_time + 5000*ms),  
        bin_size=50*ms,
        n_samples=10,
        neurons_per_sample=30,
        plot_order=plot_order,
        display_names=params.get('display_names', None),
        stimulus_config=simulation_params.get('stimulus', {}),
        save_plot=True
    )
    
    if 'voltage_monitors' in results and results['voltage_monitors']:
        plot_enhanced_multi_neuron_stimulus_overview(
            results['voltage_monitors'],
            results['spike_monitors'],
            simulation_params.get('stimulus', {}),
            target_groups=['FSN', 'MSND1', 'MSND2', 'GPeT1', 'GPeTA', 'STN', 'GPi', 'SNr'],
            neurons_per_group=2,
            analysis_window=(analysis_start_time, min(analysis_end_time, analysis_start_time + 5000*ms)),
            unified_y_scale=True,
            threshold_clipping=True,
            display_names=params.get('display_names', None),
            save_plot=True
        )
    else:
        print("Voltage monitors not available - skipping membrane potential analysis")

    try:
        pass
    
    except Exception as e:
        print(f"Error during spike interval analysis: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()