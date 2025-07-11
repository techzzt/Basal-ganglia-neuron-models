#!/usr/bin/env python3
import json
import numpy as np
from module.simulation.runner import run_simulation_with_inh_ext_input
from module.utils.param_loader import load_params
from module.utils.sta import compute_isi_all_neurons, debug_isi_data
from brian2 import ms 

def main():
    params_file = 'config/test_normal_noin.json'
    
    # 1. 첫 번째 실험 (normal 조건): save_isi_ranges=True, use_saved_ranges=False
    #    -> 현재 실험의 축 범위를 파일에 저장
    # 2. 두 번째 실험 (PD 조건): save_isi_ranges=False, use_saved_ranges=True  
    #    -> 저장된 축 범위를 불러와서 동일한 축 범위로 그래프 생성
    
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
    
    # Print actual stimulus timing if enabled
    stimulus_enabled = simulation_params.get('stimulus', {}).get('enabled', False)
    if stimulus_enabled:
        stimulus_config = simulation_params['stimulus']
        stim_start_time = stimulus_config.get('start_time', 0)
        stim_duration = stimulus_config.get('duration', 0)
        stim_end_time = stim_start_time + stim_duration
        print(f"Stimulus: {stim_start_time}-{stim_end_time}ms")
    else:
        print("Stimulus: disabled")
       
    try:
        print("\n--- Full Period Analysis ---")
        
        # Debug: Check ISI data status
        debug_isi_data(results['spike_monitors'], analysis_start_time, analysis_end_time)
        
        isi_results = compute_isi_all_neurons(
            results['spike_monitors'], 
            start_time=analysis_start_time, 
            end_time=analysis_end_time, 
            plot_order=plot_order,
            plot_histograms=True,
            display_names=params.get('display_names', None),
            use_saved_ranges=use_saved_ranges,
            save_ranges=save_isi_ranges,
            ranges_filename=ranges_filename
        )
        
        if stimulus_enabled:
            stim_start = stimulus_config.get('start_time', 10000) * ms
            stim_duration = stimulus_config.get('duration', 1000) * ms
            stim_end = stim_start + stim_duration
            window = 1000 * ms 
            
            pre_start = stim_start - window
            pre_end = stim_start
            print(f"\n--- Pre-stimulus Period ({pre_start/ms:.0f}-{pre_end/ms:.0f}ms) ---")
            compute_isi_all_neurons(
                results['spike_monitors'], 
                start_time=pre_start, 
                end_time=pre_end, 
                plot_order=plot_order,
                display_names=params.get('display_names', None)
            )
            
            print(f"\n--- During Stimulus Period ({stim_start/ms:.0f}-{stim_end/ms:.0f}ms) ---")
            compute_isi_all_neurons(
                results['spike_monitors'], 
                start_time=stim_start, 
                end_time=stim_end, 
                plot_order=plot_order,
                display_names=params.get('display_names', None)
            )
            
            post_start = stim_end
            post_end = stim_end + window
            print(f"\n--- Post-stimulus Period ({post_start/ms:.0f}-{post_end/ms:.0f}ms) ---")
            compute_isi_all_neurons(
                results['spike_monitors'], 
                start_time=post_start, 
                end_time=post_end, 
                plot_order=plot_order,
                display_names=params.get('display_names', None)
            )
    
    except Exception as e:
        print(f"Error during spike interval analysis: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()