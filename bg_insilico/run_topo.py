#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) 2025. All rights reserved.
# Author: keun (Jieun Kim)

import json
import numpy as np
import os
import argparse
import pickle
from brian2 import *
from brian2 import prefs
from datetime import datetime
prefs.codegen.target = 'numpy'

from module.utils.param_loader import load_params
from module.utils.visualization import (
    plot_raster_random_regions,
    plot_poster_raster,
    plot_stn_region_separated_raster,
)
from module.utils.firing_rate_analysis import (
    calculate_and_print_firing_rates,
    analyze_input_rates_and_spike_counts,
    analyze_firing_rates_by_stimulus_periods
)
from module.utils.stn_burst_scan import scan_stn_burst, suggest_stn_burst_params
from module.utils.topographic import TopographicMapper, should_apply_topographic
from module.models.Synapse_topo import create_synapses_with_topography
from module.utils.dopamine import apply_synaptic_dopamine
from module.models.neuron_models import create_neurons
from module.models.stimulus import create_poisson_inputs
from module.models.thalamus import Thalamus


def run_simulation_with_topography(
    neuron_configs,
    connections,
    synapse_class,
    simulation_params,
    plot_order=None,
    start_time=0*ms,
    end_time=1000*ms,
    ext_inputs=None,
    amplitude_oscillations=None,
    overlap_ratio=0.3,
    distant_ratio=0.1,
    enable_topographic=True,
    enable_thalamus=False,
    thalamus_params=None
):
    spike_monitors = {}
    neuron_groups = None
    synapse_connections = None
    poisson_groups = None
    topo_mapper = None
    results = {'spike_monitors': {}, 'firing_rates': {}}
    net = None
    duration = None
    
    try:
        try:
            from brian2.devices.device import reset_device
            reset_device()
        except:
            pass
        
        device.reinit()
        device.activate()
        
        net = Network()
        
        if enable_topographic:
            topo_mapper = TopographicMapper(
                overlap_ratio=overlap_ratio, 
                distant_ratio=distant_ratio
            )
        
        neuron_groups = create_neurons(neuron_configs, simulation_params, connections, topo_mapper=topo_mapper)
        scaled_neuron_counts = {name: group.N for name, group in neuron_groups.items()}
        poisson_groups, timed_arrays = create_poisson_inputs(
            neuron_groups, 
            ext_inputs, 
            scaled_neuron_counts,
            neuron_configs,
            amplitude_oscillations,
            simulation_params.get('stimulus', {}),
            simulation_params,
            topo_mapper=topo_mapper if enable_topographic else None,
            enable_topographic=enable_topographic
        ) if ext_inputs else ({}, [])
        
        # Create synapses with topographic organization
        all_groups = {**neuron_groups, **poisson_groups}
        
        synapse_connections, synapse_map, topo_mapper = create_synapses_with_topography(
            all_groups, connections, synapse_class,
            topo_mapper=topo_mapper,
            enable_topographic=enable_topographic,
            dop_cfg=simulation_params.get('dopamine', {}),
            distance_cache=simulation_params.get('distance_cache', {})
        )
        # Apply dopamine scaling to synapses (weights) after creation
        try:
            apply_synaptic_dopamine(synapse_map, connections, simulation_params.get('dopamine', {}))
        except Exception:
            pass
        
        voltage_monitors = {}
        mon_cfg = simulation_params.get('monitor', {}) if isinstance(simulation_params, dict) else {}
        mon_enabled = bool(mon_cfg.get('enabled', True))
        mon_record_per_group = int(mon_cfg.get('record_per_group', 3))
        mon_variables = mon_cfg.get('variables', ['v'])
        if isinstance(mon_variables, str):
            mon_variables = [mon_variables]
        mon_state_dt_ms = float(mon_cfg.get('state_dt_ms', 5.0))

        # Special monitor for STN region-specific neurons
        stn_region_monitor = None
        
        for name, group in neuron_groups.items():
            if not name.startswith(('Cortex_', 'Ext_')):
                spike_monitors[name] = SpikeMonitor(group)
                if mon_enabled:
                    num_to_record = min(mon_record_per_group, int(group.N))
                    neurons_to_record = list(range(num_to_record))
                    try:
                        voltage_monitors[name] = StateMonitor(
                            group,
                            mon_variables,
                            record=neurons_to_record,
                            dt=mon_state_dt_ms * ms
                        )
                    except Exception:
                        # fallback to only voltage if variables mismatch
                        voltage_monitors[name] = StateMonitor(
                            group,
                            'v',
                            record=neurons_to_record,
                            dt=mon_state_dt_ms * ms
                        )
        
        # Add STN region-specific voltage monitor and sample indices
        stn_region_sample_indices = {'motor': [], 'associative': [], 'limbic': []}
        if 'STN' in neuron_groups and topo_mapper is not None and enable_topographic:
            stn_group = neuron_groups['STN']
            try:
                if 'STN' not in topo_mapper.region_map:
                    topo_mapper.divide_population('STN', int(stn_group.N))
                
                motor_indices = topo_mapper.get_region_indices('STN', 0)
                asso_indices = topo_mapper.get_region_indices('STN', 1)
                limb_indices = topo_mapper.get_region_indices('STN', 2)
                
                np.random.seed(42)
                n_samples_per_region = 15
                
                if len(motor_indices) > 0:
                    motor_sampled = np.random.choice(motor_indices, min(n_samples_per_region, len(motor_indices)), replace=False)
                    stn_region_sample_indices['motor'] = np.sort(motor_sampled).tolist()
                    motor_neuron = motor_indices[len(motor_indices) // 2]
                else:
                    motor_neuron = None
                    
                if len(asso_indices) > 0:
                    asso_sampled = np.random.choice(asso_indices, min(n_samples_per_region, len(asso_indices)), replace=False)
                    stn_region_sample_indices['associative'] = np.sort(asso_sampled).tolist()
                    asso_neuron = asso_indices[len(asso_indices) // 2]
                else:
                    asso_neuron = None
                    
                if len(limb_indices) > 0:
                    limb_sampled = np.random.choice(limb_indices, min(n_samples_per_region, len(limb_indices)), replace=False)
                    stn_region_sample_indices['limbic'] = np.sort(limb_sampled).tolist()
                    limb_neuron = limb_indices[len(limb_indices) // 2]
                else:
                    limb_neuron = None
                
                stn_region_neurons = []
                for n in [motor_neuron, asso_neuron, limb_neuron]:
                    if n is not None:
                        stn_region_neurons.append(n)
                
                if len(stn_region_neurons) > 0:
                    stn_region_monitor = StateMonitor(
                        stn_group,
                        'v',
                        record=stn_region_neurons,
                        dt=mon_state_dt_ms * ms
                    )
                    net.add(stn_region_monitor)
            except Exception:
                pass
        
        # Add SNr region-specific voltage monitor (one representative per region)
        snr_region_monitor = None
        if 'SNr' in neuron_groups and topo_mapper is not None and enable_topographic:
            snr_group = neuron_groups['SNr']
            try:
                if 'SNr' not in topo_mapper.region_map:
                    topo_mapper.divide_population('SNr', int(snr_group.N))
                motor_indices = topo_mapper.get_region_indices('SNr', 0)
                asso_indices = topo_mapper.get_region_indices('SNr', 1)
                limb_indices = topo_mapper.get_region_indices('SNr', 2)
                reps = []
                if len(motor_indices) > 0:
                    reps.append(motor_indices[len(motor_indices) // 2])
                if len(asso_indices) > 0:
                    reps.append(asso_indices[len(asso_indices) // 2])
                if len(limb_indices) > 0:
                    reps.append(limb_indices[len(limb_indices) // 2])
                if len(reps) > 0:
                    snr_region_monitor = StateMonitor(
                        snr_group,
                        'v',
                        record=reps,
                        dt=mon_state_dt_ms * ms
                    )
                    net.add(snr_region_monitor)
            except Exception:
                pass

        poisson_monitors = {name: SpikeMonitor(group) for name, group in poisson_groups.items()}
        
        # Initialize Thalamus if enabled
        thalamus = None
        snr_rate_monitor = None
        snr_spike_monitor = None
        if enable_thalamus:
            thalamus_params = thalamus_params or {'mu_max': 1000*Hz, 'mu_snr_max': 2000*Hz}
            thalamus = Thalamus(**thalamus_params)
            
            # Monitor SNr firing rate for Thalamus input (both population and spike monitors for region-specific)
            if 'SNr' in neuron_groups:
                snr_rate_monitor = PopulationRateMonitor(neuron_groups['SNr'])
                snr_spike_monitor = spike_monitors.get('SNr') 
                net.add(snr_rate_monitor)
    
        net.add(*neuron_groups.values())
        net.add(*synapse_connections)
        net.add(*poisson_groups.values())
        net.add(*spike_monitors.values())
        net.add(*voltage_monitors.values())
        net.add(*poisson_monitors.values())
        
        duration = simulation_params.get('duration', 1000) * ms
        dt = simulation_params.get('dt', 0.1) * ms 
        defaultclock.dt = dt
        
        if enable_thalamus and thalamus and snr_rate_monitor:
            chunk_duration = 50*ms  
            total_chunks = int(duration / chunk_duration)
            
            for chunk in range(total_chunks):
                chunk_start = chunk * chunk_duration
                chunk_end = min((chunk + 1) * chunk_duration, duration)
                
                net.run(chunk_end - chunk_start, report='text', report_period=(chunk_end - chunk_start) * 0.5, namespace=timed_arrays)
                
                if len(snr_rate_monitor.rate) > 0:
                    window_samples = min(10, len(snr_rate_monitor.rate))
                    recent_rates = snr_rate_monitor.rate[-window_samples:]
                    current_snr_rate_raw = np.mean(recent_rates)
                    
                    dt_chunk_ms = float(chunk_duration / ms)
                    current_time_ms = float((chunk_start + chunk_duration) / ms)
                    
                    thalamus.update_cortex_lambda(
                        poisson_groups, 
                        current_snr_rate_raw,
                        dt_ms=dt_chunk_ms,
                        current_time_ms=current_time_ms,
                        snr_spike_monitor=snr_spike_monitor,
                        snr_neuron_group=neuron_groups.get('SNr'),
                        topo_mapper=topo_mapper if enable_topographic else None
                    )
                    try:
                        rl = thalamus.region_lambda
                        print(f"[Thalamus] SNr={float(current_snr_rate_raw/Hz):.2f} Hz → motor={float(rl['motor']/Hz):.2f}, asso={float(rl['associative']/Hz):.2f}, limbic={float(rl['limbic']/Hz):.2f} Hz")
                    except Exception:
                        pass
        else:
            net.run(duration, report='text', report_period=duration * 0.5, namespace=timed_arrays)
        
        if not spike_monitors:
            return results
        
        results = {
            'spike_monitors': spike_monitors,
            'voltage_monitors': voltage_monitors,
            'poisson_monitors': poisson_monitors,
            'neuron_groups': neuron_groups,
            'firing_rates': {},
            'synapses': synapse_map,
            'connections': connections,
            'topo_mapper': topo_mapper,
            'thalamus': thalamus,
            'snr_rate_monitor': snr_rate_monitor,
            'thalamus_lambda_history': thalamus.get_current_lambda() if thalamus else {},
            'stn_region_monitor': stn_region_monitor,
            'stn_region_sample_indices': stn_region_sample_indices,
            'snr_region_monitor': snr_region_monitor
        }
        
    except Exception as e:
        raise
    
    finally:
        if spike_monitors:
            for mon in spike_monitors.values():
                if hasattr(mon, 'active'):
                    mon.active = False
        
        for obj in [spike_monitors, neuron_groups, synapse_connections, poisson_groups]:
            if obj is not None:
                del obj
        
        try:
            device.reinit()
            device.activate()
        except:
            pass
    
    return results


def parse_arguments():
    parser = argparse.ArgumentParser(
        description='Run topographic basal ganglia network simulation',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
        Examples:
        python run_topo.py
        python run_topo.py --config config/test_normal_noin.json
        python run_topo.py --config config/test_dop.json --overlap-ratio 0.4
        python run_topo.py --no-topographic 
        """
    )
    parser.add_argument('--config', '-c', 
                       default='config/test_normal_noin.json',
                       help='Path to configuration JSON file')
    parser.add_argument('--list-configs', '-l', 
                       action='store_true',
                       help='List available configuration files')
    parser.add_argument('--overlap-ratio', type=float, default=0.3,
                       help='Connection ratio for adjacent regions (default: 0.3)')
    parser.add_argument('--distant-ratio', type=float, default=0.1,
                       help='Connection ratio for distant regions (default: 0.1)')
    parser.add_argument('--no-topographic', action='store_true',
                       help='Disable topographic organization (use standard connectivity)')
    parser.add_argument('--save-prefix', type=str, default='topo',
                       help='Prefix for saved files (default: topo)')
    return parser.parse_args()


def list_available_configs():
    config_dir = 'config'
    if os.path.exists(config_dir):
        config_files = [f for f in os.listdir(config_dir) if f.endswith('.json')]
        print("Available configuration files:")
        for i, config in enumerate(sorted(config_files), 1):
            print(f"  {i}. {config}")
        print(f"\nUse: python run_topo.py --config config/{config_files[0] if config_files else 'your_config.json'}")
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
    
    params = load_params(params_file)
    neuron_configs = params['neurons']
    connections = params['connections']   
    synapse_class = params.get('synapse_class', 'Synapse')
    simulation_params = params['simulation']

    if 'dopamine' not in simulation_params:
        simulation_params['dopamine'] = {
            'enabled': False,
            'alpha_dop': 0.8,
            'alpha0': 0.8,
            'EL_delta_mV': 10.0 
        }
    plot_order = params['plot_order']
    analysis_start_time = params.get('start_time', 2000) * ms
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
    
    # Thalamus parameters (config override)
    thal_cfg = simulation_params.get('thalamus', {}) if isinstance(simulation_params, dict) else {}
    mu_max_hz = thal_cfg.get('mu_max_hz', thal_cfg.get('mu_max', None))
    mu_snr_max_hz = thal_cfg.get('mu_snr_max_hz', thal_cfg.get('mu_snr_max', None))
    mu_min_hz = thal_cfg.get('mu_min_hz', thal_cfg.get('mu_min', None))
    snr_min_hz = thal_cfg.get('snr_min_hz', thal_cfg.get('snr_min', None))
    filter_tau_ms = thal_cfg.get('filter_tau_ms', None)
    window_ms_cfg = thal_cfg.get('window_ms', None)

    thalamus_params = {
        'mu_max': (mu_max_hz * Hz) if isinstance(mu_max_hz, (int, float)) else 40*Hz,
        'mu_snr_max': (mu_snr_max_hz * Hz) if isinstance(mu_snr_max_hz, (int, float)) else 35*Hz
    }
    if isinstance(mu_min_hz, (int, float)):
        thalamus_params['mu_min'] = mu_min_hz * Hz
    else:
        thalamus_params['mu_min'] = 10*Hz
    if isinstance(snr_min_hz, (int, float)):
        thalamus_params['snr_min'] = snr_min_hz * Hz
    if isinstance(filter_tau_ms, (int, float)):
        thalamus_params['filter_tau'] = filter_tau_ms * ms
    if isinstance(window_ms_cfg, (int, float)):
        thalamus_params['window_ms'] = float(window_ms_cfg)
    
    results = run_simulation_with_topography(
        neuron_configs=neuron_configs,
        connections=connections,
        synapse_class=synapse_class,
        simulation_params=simulation_params,    
        plot_order=plot_order, 
        start_time=analysis_start_time,
        end_time=analysis_end_time,
        ext_inputs=ext_inputs,
        amplitude_oscillations=amplitude_oscillations,
        overlap_ratio=args.overlap_ratio,
        distant_ratio=args.distant_ratio,
        enable_topographic=(not args.no_topographic),
        enable_thalamus=True,
        thalamus_params=thalamus_params
    )
    
    try:
        calculate_and_print_firing_rates(
            results['spike_monitors'], 
            start_time=analysis_start_time,
            end_time=analysis_end_time,
            display_names=params.get('display_names', None),
            bin_size=100*ms
        )
    except Exception as e:
        print(f"Warning: firing rate analysis failed: {e}")
    
    # Analyze input rates and spike counts
    try:
        analyze_input_rates_and_spike_counts(
            results['spike_monitors'],
            ext_inputs,
            neuron_configs,
            simulation_params.get('stimulus', {}),
            analysis_start_time,
            analysis_end_time
        )
    except Exception as e:
        print(f"Warning: input rate/spike count analysis failed: {e}")
    
    # Analyze firing rates by stimulus periods
    try:
        analyze_firing_rates_by_stimulus_periods(
            results['spike_monitors'],
            simulation_params.get('stimulus', {}),
            analysis_start_time,
            plot_order,
            params.get('display_names', None)
        )
    except Exception as e:
        print(f"Warning: stimulus-period firing rate analysis failed: {e}")

    # STN burst parameter scan
    try:
        stn_base = {
            'a': 0.3e-9,
            'd': 0.05e-12,
            'C': 60e-12,
            'Delta_T': 16.2e-3,
            'E_L': -80.2e-3,
            'g_L': 10e-9,
            'I_ext': 5e-12,
            'tau_w': 333e-3,
            'vt': -64e-3,
            'vr': -70e-3,
            'v': -75e-3,
        }
        scan_results = scan_stn_burst(stn_base, duration_ms=2000.0)
        ps, bf, rate = suggest_stn_burst_params(scan_results, target_burst=0.5)
        if ps:
            print(f"\n[STN burst suggestion] target≈50% → a={ps['a']:.2e} S, tau_w={ps['tau_w']:.2e} s, "
                  f"Delta_T={ps['Delta_T']:.2e} V, E_L={ps['E_L']:.2e} V, I_ext={ps['I_ext']:.2e} A"
                  f" | burst_fraction={bf:.2f}, rate={rate:.2f} Hz")
        else:
            print("\n[STN burst suggestion] no result")
    except Exception as e:
        print(f"STN burst scan failed: {e}")
    
    prefix = args.save_prefix
    if prefix == 'topo':
        try:
            base_name = os.path.splitext(os.path.basename(params_file))[0]
        except Exception:
            base_name = 'config'
        topo_tag = 'topo' if not args.no_topographic else 'noTopo'
        codegen_tag = getattr(prefs, 'codegen', None)
        target_tag = getattr(codegen_tag, 'target', 'numpy') if codegen_tag is not None else 'numpy'
        prefix = f"{base_name}_{topo_tag}_{target_tag}"

    # Use a dedicated timestamped result directory for all outputs
    base_result_dir = 'result'
    try:
        os.makedirs(base_result_dir, exist_ok=True)
    except Exception:
        pass
    timestamp = datetime.now().strftime('%Y%m%d_%H%M')
    output_dir = os.path.join(base_result_dir, timestamp)
    try:
        os.makedirs(output_dir, exist_ok=True)
    except Exception:
        pass
    
    try:
        if 'topo_mapper' in results and results['topo_mapper'] is not None:
            plot_poster_raster(
                results['spike_monitors'],
                results['topo_mapper'],
                results['neuron_groups'],
                n_per_region=25,
                start_time=analysis_start_time,
                end_time=analysis_end_time,
                display_names=params.get('display_names', None),
                save_png=True,
                png_filename=os.path.join(output_dir, f'{prefix}_poster_raster.png'),
                save_eps=True,
                eps_filename=os.path.join(output_dir, f'{prefix}_poster_raster.eps')
            )
            
            # Also generate region-separated raster plot
            plot_raster_random_regions(
                results['spike_monitors'],
                n_per_region=15,
                plot_order=plot_order,
                start_time=analysis_start_time,
                end_time=analysis_end_time,
                display_names=params.get('display_names', None),
                seed=42,
                save_plot=True,
                save_png=True,
                png_filename=os.path.join(output_dir, f'{prefix}_raster_random_regions.png'),
                save_eps=True,
                eps_filename=os.path.join(output_dir, f'{prefix}_raster_random_regions.eps'),
                topo_mapper=results['topo_mapper']
            )
    except Exception as e:
        print(f"Error in raster plot: {e}")
    
    try:
        if 'stn_region_monitor' in results and results['stn_region_monitor'] is not None:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            monitor = results['stn_region_monitor']
            
            try:
                t_data = np.array(monitor.t) if hasattr(monitor.t, '__iter__') else monitor.t
                v_data = monitor.v
                
                if len(t_data) > 0:
                    t_ms = np.array(t_data / ms)
                    
                    if hasattr(v_data, '__iter__'):
                        v_arr = np.array(v_data / mV)
                        if v_arr.ndim == 1:
                            v_arr = v_arr.reshape(1, -1)
                    else:
                        v_arr = np.array([v_data / mV]).reshape(1, -1)
                    
                    if v_arr.ndim == 2 and v_arr.shape[0] >= 1:
                        n_neurons = v_arr.shape[0]
                        fig, axes = plt.subplots(n_neurons, 1, figsize=(12, 3 * n_neurons), sharex=True)
                        if n_neurons == 1:
                            axes = [axes]
                        
                        region_names = ['Motor', 'Associative', 'Limbic']
                        colors = ['#d62728', '#1f77b4', '#2ca02c']
                        
                        for i in range(n_neurons):
                            region_name = region_names[i] if i < len(region_names) else f'Region {i}'
                            color = colors[i] if i < len(colors) else 'black'
                            neuron_idx = monitor.record[i] if hasattr(monitor, 'record') and i < len(monitor.record) else i
                            
                            if len(t_ms) == v_arr.shape[1]:
                                axes[i].plot(t_ms, v_arr[i, :], color=color, linewidth=1.0, alpha=0.8)
                            else:
                                min_len = min(len(t_ms), v_arr.shape[1])
                                axes[i].plot(t_ms[:min_len], v_arr[i, :min_len], color=color, linewidth=1.0, alpha=0.8)
                            
                            axes[i].set_ylabel(f'{region_name}\nVoltage (mV)', fontsize=10)
                            axes[i].grid(True, alpha=0.3)
                            axes[i].set_title(f'STN {region_name} Region (Neuron {neuron_idx})', fontsize=11, fontweight='bold')
                        
                        axes[-1].set_xlabel('Time (ms)', fontsize=12)
                        plt.tight_layout()
                        
                        png_filename = os.path.join(output_dir, f'{prefix}_stn_region_membrane_potential.png')
                        eps_filename = os.path.join(output_dir, f'{prefix}_stn_region_membrane_potential.eps')
                        plt.savefig(png_filename, dpi=300, bbox_inches='tight', facecolor='white')
                        plt.savefig(eps_filename, format='eps', dpi=300, bbox_inches='tight', facecolor='white')
                        plt.close()
            except Exception:
                pass
    except Exception:
        pass
    
    try:
        from module.utils.lfp_visualization import (
            plot_region_overlaid_lfp,
            plot_region_specific_lfp_type1,
            plot_region_specific_lfp_type2,
            plot_stn_region_lfp_overlaid
        )
        
        if 'topo_mapper' in results and results['topo_mapper'] is not None:
            plot_region_overlaid_lfp(
                results['spike_monitors'],
                results['topo_mapper'],
                results['neuron_groups'],
                start_time=analysis_start_time,
                end_time=analysis_end_time,
                target_groups=['STN', 'SNr', 'GPeT1', 'GPeTA'],
                n_samples_per_region=30,
                save_png=True,
                png_filename=os.path.join(output_dir, f'{prefix}_region_overlaid_lfp.png'),
                save_eps=True,
                eps_filename=os.path.join(output_dir, f'{prefix}_region_overlaid_lfp.eps'),
                seed=42
            )
            
            plot_region_specific_lfp_type1(
                results['spike_monitors'],
                results['topo_mapper'],
                results['neuron_groups'],
                start_time=analysis_start_time,
                end_time=analysis_end_time,
                target_groups=['STN', 'SNr', 'GPeT1', 'GPeTA'],
                n_samples_per_region=15,
                save_png=True,
                png_filename=os.path.join(output_dir, f'{prefix}_region_lfp_type1.png'),
                save_eps=True,
                eps_filename=os.path.join(output_dir, f'{prefix}_region_lfp_type1.eps'),
                seed=42
            )
            
            plot_region_specific_lfp_type2(
                results['spike_monitors'],
                results['topo_mapper'],
                results['neuron_groups'],
                start_time=analysis_start_time,
                end_time=analysis_end_time,
                target_groups=['STN', 'SNr', 'GPeT1', 'GPeTA'],
                n_samples_per_region=15,
                save_png=True,
                png_filename=os.path.join(output_dir, f'{prefix}_region_lfp_type2.png'),
                save_eps=True,
                eps_filename=os.path.join(output_dir, f'{prefix}_region_lfp_type2.eps'),
                seed=42
            )
            
            if 'stn_region_sample_indices' in results and results['stn_region_sample_indices']:
                plot_stn_region_lfp_overlaid(
                    results['spike_monitors'],
                    results['stn_region_sample_indices'],
                    start_time=analysis_start_time,
                    end_time=analysis_end_time,
                    save_png=True,
                    png_filename=os.path.join(output_dir, f'{prefix}_stn_region_lfp_overlaid.png'),
                    save_eps=True,
                    eps_filename=os.path.join(output_dir, f'{prefix}_stn_region_lfp_overlaid.eps')
                )
            
            if 'stn_region_sample_indices' in results and results['stn_region_sample_indices']:
                plot_stn_region_separated_raster(
                    results['spike_monitors'],
                    results['stn_region_sample_indices'],
                    start_time=analysis_start_time,
                    end_time=analysis_end_time,
                    save_png=True,
                    png_filename=os.path.join(output_dir, f'{prefix}_stn_region_separated_raster.png'),
                    save_eps=True,
                    eps_filename=os.path.join(output_dir, f'{prefix}_stn_region_separated_raster.eps')
                )
    except Exception:
        pass
    
    # Save pickle under timestamped result directory
    save_filename = os.path.join(output_dir, f'{prefix}_results.pkl')
    
    try:
        save_data = {
            'meta': {
                'config': os.path.basename(params_file),
                'start_time_ms': int(analysis_start_time/ms),
                'end_time_ms': int(analysis_end_time/ms),
                'default_bin_size_ms': 10,
                'topographic_enabled': not args.no_topographic,
                'overlap_ratio': args.overlap_ratio,
                'distant_ratio': args.distant_ratio
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
        
        if 'topo_mapper' in results and results['topo_mapper'] is not None:
            save_data['topographic_regions'] = results['topo_mapper'].region_map
        
        if 'stn_region_monitor' in results and results['stn_region_monitor'] is not None:
            monitor = results['stn_region_monitor']
            try:
                t_data = np.array(monitor.t) if hasattr(monitor.t, '__iter__') else monitor.t
                v_data = monitor.v
                if len(t_data) > 0:
                    t_ms_arr = np.array(t_data / ms, dtype=float)
                    if hasattr(v_data, '__iter__'):
                        v_arr = np.array(v_data / mV, dtype=float)
                        if v_arr.ndim == 1:
                            v_arr = v_arr.reshape(1, -1)
                    else:
                        v_arr = np.array([v_data / mV], dtype=float).reshape(1, -1)
                    
                    if v_arr.ndim == 2:
                        save_data['stn_region_monitor'] = {
                            't_ms': t_ms_arr,
                            'v_mV': v_arr,
                            'record': monitor.record if hasattr(monitor, 'record') else list(range(v_arr.shape[0])),
                            'N': v_arr.shape[0]
                        }
            except Exception:
                pass
                
        # Save SNr region-specific monitor if available
        if 'snr_region_monitor' in results and results['snr_region_monitor'] is not None:
            monitor = results['snr_region_monitor']
            try:
                t_data = np.array(monitor.t) if hasattr(monitor.t, '__iter__') else monitor.t
                v_data = monitor.v
                if len(t_data) > 0:
                    t_ms_arr = np.array(t_data / ms, dtype=float)
                    if hasattr(v_data, '__iter__'):
                        v_arr = np.array(v_data / mV, dtype=float)
                        if v_arr.ndim == 1:
                            v_arr = v_arr.reshape(1, -1)
                    else:
                        v_arr = np.array([v_data / mV], dtype=float).reshape(1, -1)
                    if v_arr.ndim == 2:
                        save_data['snr_region_monitor'] = {
                            't_ms': t_ms_arr,
                            'v_mV': v_arr,
                            'record': monitor.record if hasattr(monitor, 'record') else list(range(v_arr.shape[0])),
                            'N': v_arr.shape[0]
                        }
            except Exception:
                pass
        
        if 'plot_order' in params:
            save_data['plot_order'] = params['plot_order']
        if 'display_names' in params:
            save_data['display_names'] = params['display_names']
        
        if 'stn_region_sample_indices' in results and results['stn_region_sample_indices']:
            save_data['stn_region_sample_indices'] = results['stn_region_sample_indices']
        
        with open(save_filename, 'wb') as f:
            pickle.dump(save_data, f)
        try:
            print(f"Pickle saved: {save_filename}")
        except Exception:
            pass
    
    except Exception as e:
        try:
            print(f"Error saving pickle to {save_filename}: {e}")
        except Exception:
            pass


if __name__ == "__main__":
    main()

