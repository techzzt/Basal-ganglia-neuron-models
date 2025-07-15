#!/usr/bin/env python3
import json
import numpy as np
from module.simulation.runner import run_simulation_with_inh_ext_input
from module.utils.param_loader import load_params
from module.utils.sta import compute_isi_all_neurons, debug_isi_data
from module.utils.visualization import (analyze_firing_rates_by_stimulus_periods, plot_continuous_firing_rate, 
                                       plot_membrane_potential, plot_beta_oscillation_analysis, plot_lfp_like_analysis,
                                       plot_separated_network_connectivity_analysis, plot_separated_spike_propagation_analysis,
                                       plot_grouped_raster_by_target, plot_stimulus_zoom_raster, plot_improved_overall_raster)
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
        
        # 스티뮬러스 구간별 발화율 분석
        analyze_firing_rates_by_stimulus_periods(
            results['spike_monitors'],
            stimulus_config,
            analysis_start_time,
            plot_order,
            params.get('display_names', None)
        )
        
        # 연속적인 발화율 변화 그래프 (전문적 스타일)
        plot_continuous_firing_rate(
            results['spike_monitors'],
            start_time=analysis_start_time,
            end_time=analysis_end_time,
            bin_size=20*ms,  # 20ms 구간으로 더 세밀하게 계산
            plot_order=plot_order,
            display_names=params.get('display_names', None),
            stimulus_config=stimulus_config,
            smooth_sigma=3,  # Gaussian smoothing
            show_confidence=True  # 신뢰구간 표시
        )
        
        # Beta oscillation specialized analysis
        print("\n=== BETA OSCILLATION ANALYSIS START ===")
        plot_beta_oscillation_analysis(
            results['spike_monitors'],
            start_time=analysis_start_time,
            end_time=analysis_end_time,
            plot_order=plot_order,
            display_names=params.get('display_names', None),
            stimulus_config=stimulus_config,
            save_plot=True
        )
        
        # Beta-focused LFP analysis
        print("\n=== BETA-FOCUSED LFP ANALYSIS START ===")
        plot_lfp_like_analysis(
            results['spike_monitors'],
            start_time=analysis_start_time,
            end_time=analysis_end_time,
            plot_order=plot_order,
            display_names=params.get('display_names', None),
            stimulus_config=stimulus_config,
            save_plot=True
        )
        
        # Network connectivity analysis (FSN-centered)
        print("\n=== FSN NETWORK CONNECTIVITY ANALYSIS START ===")
        plot_network_connectivity_analysis(
            results['spike_monitors'],
            connections,  # Pass connection configuration
            plot_order=plot_order,
            display_names=params.get('display_names', None),
            target_neuron='FSN',
            time_window=50*ms,
            save_plot=True
        )
        
        # Spike propagation analysis (FSN-centered, stimulus period)
        print("\n=== FSN SPIKE PROPAGATION ANALYSIS START ===")
        plot_spike_propagation_analysis(
            results['spike_monitors'],
            connections,
            target_neuron='FSN',
            analysis_window=(stim_start_time*ms, stim_end_time*ms),
            propagation_delay=20*ms,
            save_plot=True
        )
        
    else:
        print("Stimulus: disabled")
        
        # Run beta oscillation analysis even without stimulus
        print("\n=== BETA OSCILLATION ANALYSIS START (No Stimulus) ===")
        plot_beta_oscillation_analysis(
            results['spike_monitors'],
            start_time=analysis_start_time,
            end_time=analysis_end_time,
            plot_order=plot_order,
            display_names=params.get('display_names', None),
            stimulus_config=None,
            save_plot=True
        )
        
        print("\n=== BETA-FOCUSED LFP ANALYSIS START (No Stimulus) ===")
        plot_lfp_like_analysis(
            results['spike_monitors'],
            start_time=analysis_start_time,
            end_time=analysis_end_time,
            plot_order=plot_order,
            display_names=params.get('display_names', None),
            stimulus_config=None,
            save_plot=True
        )
        
        # Separated Network connectivity analysis (post neuron별로 분리)
        print("\n=== SEPARATED NETWORK CONNECTIVITY ANALYSIS START ===")
        plot_separated_network_connectivity_analysis(
            results['spike_monitors'],
            connections,
            plot_order=plot_order,
            display_names=params.get('display_names', None),
            time_window=50*ms,
            save_plot=True
        )
        
        # Separated Spike propagation analysis (post neuron별로 분리)
        print("\n=== SEPARATED SPIKE PROPAGATION ANALYSIS START ===")
        plot_separated_spike_propagation_analysis(
            results['spike_monitors'],
            connections,
            analysis_window=(analysis_start_time, analysis_start_time + 2000*ms),  # Analyze first 2 seconds
            propagation_delay=20*ms,
            save_plot=True
        )
        
    # Enhanced Raster Plot Visualizations
    print("\n=== ENHANCED RASTER PLOT ANALYSIS START ===")
    
    # 1. Individual group raster plots (to reduce overlapping)
    print("\n--- Individual Group Raster Plots ---")
    plot_grouped_raster_by_target(
        results['spike_monitors'],
        sample_size=8,  # Very small sample to prevent overlap
        plot_order=plot_order,
        start_time=analysis_start_time,
        end_time=analysis_end_time,
        display_names=params.get('display_names', None),
        save_plot=True
    )
    
    # 2. Improved overall raster plot with better spacing
    print("\n--- Improved Overall Raster Plot ---")
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
        sample_size=12,  # Reduced sample size for clarity
        plot_order=plot_order,
        start_time=analysis_start_time,
        end_time=analysis_end_time,
        display_names=params.get('display_names', None),
        stimulus_periods=stimulus_periods,
        save_plot=True
    )
    
    # 3. Stimulus period zoom raster plots (if stimulus periods exist)
    if stimulus_periods:
        print("\n--- Stimulus Period Zoom Raster Plots ---")
        plot_stimulus_zoom_raster(
            results['spike_monitors'],
            stimulus_periods,
            sample_size=6,  # Very small sample for detailed zoom view
            zoom_margin=50*ms,  # 50ms margin around stimulus
            plot_order=plot_order,
            display_names=params.get('display_names', None),
            save_plot=True
        )
       
    try:
        # ISI 분석 관련 코드들은 주석처리 (평균 발화율 분석만 출력)
        # print("\n--- Full Period Analysis ---")
        # debug_isi_data(results['spike_monitors'], analysis_start_time, analysis_end_time)
        # isi_results = compute_isi_all_neurons(...)
        pass
    
    except Exception as e:
        print(f"Error during spike interval analysis: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()