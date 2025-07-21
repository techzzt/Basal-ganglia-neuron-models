#!/usr/bin/env python3
import json
import numpy as np
import sys
import os

# pd 모드 감지 및 matplotlib 백엔드 설정
pd_mode = 'pd' in sys.argv or any('pd' in arg for arg in sys.argv)
if pd_mode:
    os.environ['MPLBACKEND'] = 'Agg'  # 비대화형 백엔드 설정
    print("PD mode detected - using non-interactive backend")
else:
    print("Normal mode - using interactive backend")

from module.simulation.runner import run_simulation_with_inh_ext_input
from module.utils.param_loader import load_params
from module.utils.sta import compute_isi_all_neurons, debug_isi_data
from module.utils.visualization import (analyze_firing_rates_by_stimulus_periods, plot_continuous_firing_rate, 
                                       plot_beta_oscillation_analysis,
                                       plot_grouped_raster_by_target, plot_stimulus_zoom_raster, plot_improved_overall_raster,
                                       plot_circuit_flow_heatmap, plot_spike_burst_cascade, 
                                       plot_single_neuron_detailed_analysis,
                                       plot_neuron_stimulus_pattern, plot_multi_neuron_stimulus_overview,
                                       plot_phase_space_only_overview, plot_mean_firing_rate_heatmap,
                                       plot_improved_membrane_potential, plot_enhanced_multi_neuron_stimulus_overview,
                                       plot_multi_sample_firing_rate_analysis, plot_continuous_firing_rate_with_samples,
                                       plot_individual_neuron_firing_rates, plot_place_cell_theta_analysis)
from brian2 import ms 

def main():
    params_file = 'config/test_dop_noin.json'
    
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
    
    if pd_mode:
        print("\n=== PD MODE INFORMATION ===")
        print("In PD mode, graphs are saved as files but not displayed interactively.")
        print("Check the generated PNG files in the current directory.")
        print("To see graphs interactively, run without 'pd' argument.")
    
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
        # 레이아웃 옵션: 'single' (한 페이지에 모든 그래프) 또는 'multi' (페이지당 3개씩)
        plot_continuous_firing_rate(
            results['spike_monitors'],
            start_time=analysis_start_time,
            end_time=analysis_end_time,  # 전체 시뮬레이션 시간 사용
            bin_size=20*ms,  # 20ms 구간으로 더 세밀하게 계산
            plot_order=plot_order,
            display_names=params.get('display_names', None),
            stimulus_config=stimulus_config,
            smooth_sigma=3,  # Gaussian smoothing
            show_confidence=True,  # 신뢰구간 표시
            layout_mode='multi',  # 'single' 또는 'multi' 선택
            plots_per_page=3  # 페이지당 그래프 수 (layout_mode='multi'일 때)
        )
        
        # === NEW: Continuous Firing Rate with Multi-Sampling === (비활성화 - 중복 제거)
        # print("\n=== CONTINUOUS FIRING RATE WITH MULTI-SAMPLING START ===")
        # plot_continuous_firing_rate_with_samples(
        #     results['spike_monitors'],
        #     start_time=analysis_start_time,
        #     end_time=min(analysis_end_time, analysis_start_time + 5000*ms),  # 5초간 분석
        #     bin_size=50*ms,
        #     plot_order=plot_order,
        #     display_names=params.get('display_names', None),
        #     stimulus_config=stimulus_config,
        #     smooth_sigma=3,
        #     save_plot=True,
        #     n_samples=10,
        #     neurons_per_sample=30
        # )
        
        # === NEW: Individual Neuron Firing Rates ===
        print("\n=== INDIVIDUAL NEURON FIRING RATES START ===")
        plot_individual_neuron_firing_rates(
            results['spike_monitors'],
            start_time=analysis_start_time,
            end_time=min(analysis_end_time, analysis_start_time + 3000*ms),  # 3초간 분석
            bin_size=20*ms,
            plot_order=plot_order,
            display_names=params.get('display_names', None),
            stimulus_config=stimulus_config,
            smooth_sigma=3,
            save_plot=True,
            neurons_per_group=10  # 그룹당 10개 뉴런씩 표시 (10개 개별 + 1개 평균)
        )
        
        # === NEW: Place Cell Theta Analysis ===
        print("\n=== PLACE CELL THETA ANALYSIS START ===")
        plot_place_cell_theta_analysis(
            results['spike_monitors'],
            start_time=analysis_start_time,
            end_time=min(analysis_end_time, analysis_start_time + 5000*ms),  # 5초간 분석
            plot_order=plot_order,
            display_names=params.get('display_names', None),
            stimulus_config=stimulus_config,
            save_plot=True,
            place_field_center=0.0,  # place field 중심
            spatial_range=3.0  # 공간 분석 범위
        )
        
        # Beta oscillation specialized analysis (비활성화)
        # print("\n=== BETA OSCILLATION ANALYSIS START ===")
        # plot_beta_oscillation_analysis(
        #     results['spike_monitors'],
        #     start_time=analysis_start_time,
        #     end_time=analysis_end_time,
        #     plot_order=plot_order,
        #     display_names=params.get('display_names', None),
        #     stimulus_config=stimulus_config,
        #     save_plot=True
        # )
        
        # Beta-focused LFP analysis (비활성화)
        # print("\n=== BETA-FOCUSED LFP ANALYSIS START ===")
        # plot_lfp_like_analysis(
        #     results['spike_monitors'],
        #     start_time=analysis_start_time,
        #     end_time=analysis_end_time,
        #     plot_order=plot_order,
        #     display_names=params.get('display_names', None),
        #     stimulus_config=stimulus_config,
        #     save_plot=True
        # )
        
        # === NEW: Circuit Flow Heat Map Analysis ===
        print("\n=== CIRCUIT FLOW HEAT MAP ANALYSIS START ===")
        plot_circuit_flow_heatmap(
            results['spike_monitors'],
            connections,
            start_time=analysis_start_time,
            end_time=min(analysis_end_time, analysis_start_time + 3000*ms),  # 3초간 분석
            bin_size=10*ms,  # 10ms 해상도로 더 세밀하게
            plot_order=plot_order,
            display_names=params.get('display_names', None),
            save_plot=True
        )
        
        # === NEW: Spike Burst Cascade Analysis ===
        print("\n=== SPIKE BURST CASCADE ANALYSIS START ===")
        plot_spike_burst_cascade(
            results['spike_monitors'],
            connections,  # 연결 관계 정보 추가
            start_time=analysis_start_time,
            end_time=analysis_end_time,  # 전체 구간 분석
            burst_threshold=0.75,  # 75th percentile 이상을 burst로 감지
            cascade_window=50*ms,  # 50ms 내에서 cascade 감지
            plot_order=plot_order,
            display_names=params.get('display_names', None),
            save_plot=True
        )
        
    else:
        print("Stimulus: disabled")
        
        # Run beta oscillation analysis even without stimulus (비활성화)
        # print("\n=== BETA OSCILLATION ANALYSIS START (No Stimulus) ===")
        # plot_beta_oscillation_analysis(
        #     results['spike_monitors'],
        #     start_time=analysis_start_time,
        #     end_time=analysis_end_time,
        #     plot_order=plot_order,
        #     display_names=params.get('display_names', None),
        #     stimulus_config=None,
        #     save_plot=True
        # )
        
        # Beta-focused LFP analysis (No Stimulus) - 비활성화
        # print("\n=== BETA-FOCUSED LFP ANALYSIS START (No Stimulus) ===")
        # plot_lfp_like_analysis(
        #     results['spike_monitors'],
        #     start_time=analysis_start_time,
        #     end_time=analysis_end_time,
        #     plot_order=plot_order,
        #     display_names=params.get('display_names', None),
        #     stimulus_config=None,
        #     save_plot=True
        # )
        
        # === NEW: Circuit Flow Heat Map Analysis (No Stimulus) ===
        print("\n=== CIRCUIT FLOW HEAT MAP ANALYSIS START (No Stimulus) ===")
        plot_circuit_flow_heatmap(
            results['spike_monitors'],
            connections,
            start_time=analysis_start_time,
            end_time=min(analysis_end_time, analysis_start_time + 3000*ms),  # 3초간 분석
            bin_size=10*ms,  # 10ms 해상도로 더 세밀하게
            plot_order=plot_order,
            display_names=params.get('display_names', None),
            save_plot=True
        )
        
        # === NEW: Spike Burst Cascade Analysis (No Stimulus) ===
        print("\n=== SPIKE BURST CASCADE ANALYSIS START (No Stimulus) ===")
        plot_spike_burst_cascade(
            results['spike_monitors'],
            connections,  # 연결 관계 정보 추가
            start_time=analysis_start_time,
            end_time=analysis_end_time,  # 전체 구간 분석
            burst_threshold=0.75,  # 75th percentile 이상을 burst로 감지
            cascade_window=50*ms,  # 50ms 내에서 cascade 감지
            plot_order=plot_order,
            display_names=params.get('display_names', None),
            save_plot=True
        )
    
    # === NEW: Neuron Stimulus Pattern Analysis ===
    print("\n=== NEURON-STIMULUS PATTERN ANALYSIS START ===")
    
    # Check if voltage monitors are available
    if 'voltage_monitors' in results and results['voltage_monitors']:
        # Multi-neuron stimulus overview - all groups in one image
        print("\n--- Multi-Neuron Stimulus Overview ---")
        # plot_multi_neuron_stimulus_overview(
        #     results['voltage_monitors'],
        #     results['spike_monitors'],
        #     simulation_params.get('stimulus', {}),
        #     target_groups=['FSN', 'MSND1', 'MSND2', 'GPeT1', 'GPeTA', 'STN', 'GPi', 'SNr'],
        #     neurons_per_group=2,  # Show 2 neurons per group
        #     analysis_window=(analysis_start_time, analysis_end_time),  # Full 10000ms range
        #                          display_names=params.get('display_names', None),
        #      save_plot=True
        #  )
    else:
        print("전압 모니터 데이터를 사용할 수 없습니다. Voltage monitoring이 활성화되지 않았습니다.")

    # === NEW: Single Neuron Detailed Analysis ===
    print("\n=== SINGLE NEURON DETAILED ANALYSIS START ===")
    
    # Check if voltage monitors are available
    if 'voltage_monitors' in results and results['voltage_monitors']:
        # Clean phase space only analysis for all neuron groups
        print("\n=== PHASE SPACE CLEAN OVERVIEW ===")
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
        
        # Enhanced Raster Plot Visualization - Overall View Only
    print("\n=== RASTER PLOT ANALYSIS ===")
    print("\n--- Overall Raster Plot ---")
    
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
        sample_size=10,  # Unified sample size
        plot_order=plot_order,
        start_time=analysis_start_time,
        end_time=analysis_end_time,
        display_names=params.get('display_names', None),
        stimulus_periods=stimulus_periods,
        save_plot=True
    )
       
    # === NEW: Mean Firing Rate Heatmap Analysis === (비활성화)
    # print("\n=== MEAN FIRING RATE HEATMAP ANALYSIS START ===")
    # plot_mean_firing_rate_heatmap(
    #     results['spike_monitors'],
    #     connections,
    #     start_time=analysis_start_time,
    #     end_time=min(analysis_end_time, analysis_start_time + 3000*ms),  # 3초간 분석
    #     spatial_bins=40,
    #     time_bins=60,
    #     plot_order=plot_order,
    #     display_names=params.get('display_names', None),
    #     save_plot=True
    # )
    
    # === NEW: Multi-Sample Firing Rate Analysis ===
    print("\n=== MULTI-SAMPLE FIRING RATE ANALYSIS START ===")
    plot_multi_sample_firing_rate_analysis(
        results['spike_monitors'],
        start_time=analysis_start_time,
        end_time=min(analysis_end_time, analysis_start_time + 5000*ms),  # 5초간 분석
        bin_size=50*ms,
        n_samples=10,
        neurons_per_sample=30,
        plot_order=plot_order,
        display_names=params.get('display_names', None),
        stimulus_config=simulation_params.get('stimulus', {}),
        save_plot=True
    )
    
    # === NEW: Membrane Potential Analysis ===
    if 'voltage_monitors' in results and results['voltage_monitors']:
        print("\n=== MEMBRANE POTENTIAL ANALYSIS START ===")
        # plot_improved_membrane_potential(
        #     results['voltage_monitors'],
        #     results['spike_monitors'],
        #     plot_order=plot_order,
        #     analysis_window=(analysis_start_time, analysis_end_time),  # 시뮬레이션 전체 시간 사용
        #     unified_y_scale=True,
        #     threshold_clipping=True,
        #     display_names=params.get('display_names', None),
        #     save_plot=True
        # )
        
        # Enhanced Multi-Neuron Stimulus Overview with unified y-scale
        print("\n=== ENHANCED MULTI-NEURON STIMULUS OVERVIEW START ===")
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