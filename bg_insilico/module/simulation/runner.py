# -*- coding: utf-8 -*-
# Copyright (c) 2025. All rights reserved.
# Author: keun (Jieun Kim)

import os
import sys
import time
import gc
import numpy as np

from brian2 import *
from brian2 import mV, ms, nS, Hz, second
from module.models.neuron_models import create_neurons
from module.models.Synapse import create_synapses
from module.models.stimulus import create_poisson_inputs
from module.models.thalamus import Thalamus
from brian2.devices.device import reset_device

import platform

try:
    reset_device()
except:
    pass

if platform.system() == 'Darwin':
    os.environ['CC'] = '/usr/bin/clang'
    os.environ['CXX'] = '/usr/bin/clang++'

prefs.codegen.target = 'numpy'
prefs.core.default_float_dtype = np.float32
prefs.codegen.runtime.numpy.discard_units = True
prefs.codegen.loop_invariant_optimisations = True

# Enable progress reporting
try:
    prefs.core.network.report = 'text'
except:
    pass  

# Extract spike data from monitor
def get_monitor_spikes(monitor):
    try:
        if hasattr(monitor, 't') and hasattr(monitor, 'i') and len(monitor.t) > 0:
            return monitor.t, monitor.i
        elif hasattr(monitor, '_spike_times') and hasattr(monitor, '_spike_indices'):
            return monitor._spike_times, monitor._spike_indices
        else:
            return np.array([]) * ms, np.array([])
    except:
        return np.array([]) * ms, np.array([])

# Run simulation with external inputs
def run_simulation_with_inh_ext_input(
    neuron_configs,
    connections,
    synapse_class,
    simulation_params,
    plot_order=None,
    start_time=0*ms,
    end_time=1000*ms,
    ext_inputs=None,
    amplitude_oscillations=None
):
    spike_monitors = {}
    neuron_groups = None
    synapse_connections = None
    poisson_groups = None
    net = None
    
    try:
        try:
            reset_device()
        except:
            pass
        
        device.reinit()
        device.activate()
                
        net = Network()
        neuron_groups = create_neurons(neuron_configs, simulation_params, connections)
        
        scaled_neuron_counts = {name: group.N for name, group in neuron_groups.items()}
        
        stimulus_config = simulation_params.get('stimulus', {})
        
        poisson_groups, _ = create_poisson_inputs(
            neuron_groups, 
            ext_inputs, 
            scaled_neuron_counts,
            neuron_configs,
            amplitude_oscillations,
            stimulus_config,
            simulation_params
        ) if ext_inputs else ({}, [])

        all_groups = {**neuron_groups, **poisson_groups}
        synapse_connections, synapse_map = create_synapses(all_groups, connections, synapse_class)
        
        voltage_monitors = {}
        for name, group in neuron_groups.items():
            if not name.startswith(('Cortex_', 'Ext_')):  
                spike_monitors[name] = SpikeMonitor(group)
                num_to_record = min(10, group.N)
                neurons_to_record = list(range(num_to_record))
                try:
                    voltage_monitors[name] = StateMonitor(group, ['v', 'z'], record=neurons_to_record)
                except Exception:
                    voltage_monitors[name] = StateMonitor(group, 'v', record=neurons_to_record)
        
        poisson_monitors = {name: SpikeMonitor(group) for name, group in poisson_groups.items()}
        
        # Initialize thalamus for SNr
        thalamus = None
        cortex_groups = {name: group for name, group in poisson_groups.items() if name.startswith('Cortex_')}
        snr_spike_monitor = spike_monitors.get('SNr', None)
        
        thalamus_params = simulation_params.get('thalamus', {})
        thalamus_enabled = thalamus_params.get('enabled', True)  # Default: enabled
        
        if cortex_groups and snr_spike_monitor is not None and thalamus_enabled:
            mu_max = thalamus_params.get('mu_max', 40) * Hz
            mu_snr_max = thalamus_params.get('mu_snr_max', 35) * Hz
            mu_min = thalamus_params.get('mu_min', 10) * Hz
            snr_min = thalamus_params.get('snr_min', 20) * Hz
            filter_tau = thalamus_params.get('filter_tau', 50) * ms
            window_ms = thalamus_params.get('window_ms', 300.0)
            
            # Extract cortex baselines from config (original target rates)
            cortex_baselines = {}
            for group_name, group in cortex_groups.items():
                if group_name.startswith('Cortex_'):
                    for nc in neuron_configs:
                        if nc.get('name') == group_name and 'target_rates' in nc:
                            target, rate_info = list(nc['target_rates'].items())[0]
                            rate_expr = rate_info['equation']
                            try:
                                baseline_rate = float(eval(rate_expr) / Hz) * Hz
                                cortex_baselines[group_name] = baseline_rate
                            except:
                                cortex_baselines[group_name] = mu_max
                            break

                    if group_name not in cortex_baselines:
                        cortex_baselines[group_name] = mu_max
            
            thalamus = Thalamus(mu_max=mu_max, mu_snr_max=mu_snr_max, 
                               filter_tau=filter_tau, window_ms=window_ms,
                               cortex_baselines=cortex_baselines,
                               mu_min=mu_min, snr_min=snr_min)
        
        net.add(*neuron_groups.values())
        net.add(*synapse_connections)
        net.add(*poisson_groups.values())
        net.add(*spike_monitors.values())
        net.add(*voltage_monitors.values())
        net.add(*poisson_monitors.values())
        
        defaultclock.dt = simulation_params.get('dt', 0.1) * ms
        
        duration = simulation_params.get('duration', 1000) * ms
        print(f"\n{'='*60}")
        print(f"Starting simulation (duration: {duration/ms:.0f}ms, dt: {defaultclock.dt/ms:.1f}ms)")
        print(f"{'='*60}")
        sys.stdout.flush()  
        
        start_real_time = time.time()
        update_interval = 100*ms 
        last_reported_time = 0*ms
        chunk_size = max(10*ms, update_interval)  
        total_simulated = 0*ms
        
        while total_simulated < duration:
            remaining = duration - total_simulated
            chunk = min(chunk_size, remaining)
            
            net.run(chunk, report=None)
            
            if thalamus is not None and snr_spike_monitor is not None and cortex_groups:
                window_ms = 100.0
                current_time_ms = float(total_simulated / ms)
                cutoff_ms = current_time_ms - window_ms
                
                if hasattr(snr_spike_monitor, 't') and hasattr(snr_spike_monitor, 'i') and len(snr_spike_monitor.t) > 0:
                    t_arr = np.array(snr_spike_monitor.t / ms)
                    mask = t_arr >= cutoff_ms
                    recent_spikes = np.sum(mask)
                    snr_neurons = snr_spike_monitor.source.N
                    
                    if snr_neurons > 0 and window_ms > 0:
                        snr_rate = (recent_spikes / snr_neurons) / (window_ms / 1000.0) * Hz
                    else:
                        snr_rate = 0 * Hz
                    
                    dt_ms = float(defaultclock.dt / ms)
                    thalamus.update_cortex_lambda(cortex_groups, snr_rate, dt_ms=dt_ms, current_time_ms=current_time_ms)
                    
                    if int(current_time_ms / 1000) != int((current_time_ms - chunk_size/ms) / 1000):
                        lambda_values = thalamus.get_current_lambda()
                        print(f"\n[Thalamus Debug @ {current_time_ms:.0f}ms] SNr={snr_rate/Hz:.1f}Hz, Cortex rates:")
                    
                        for name, rate in lambda_values.items():
                            print(f"  {name}: {rate/Hz:.1f}Hz")
                    
                        sys.stdout.flush()
            
            total_simulated += chunk
            current_real_time = time.time()
            elapsed_real_time = current_real_time - start_real_time
            
            # Report progress if enough simulation time has passed (update_interval)
            if total_simulated - last_reported_time >= update_interval or total_simulated >= duration:
                progress_pct = (total_simulated / duration) * 100
                
                elapsed_min = int(elapsed_real_time // 60)
                elapsed_sec = int(elapsed_real_time % 60)
                
                if elapsed_real_time > 0 and progress_pct > 0:
                    estimated_total_time = elapsed_real_time / (progress_pct / 100.0)
                    estimated_remaining = estimated_total_time - elapsed_real_time
                    
                    remaining_hr = int(estimated_remaining // 3600)
                    remaining_min = int((estimated_remaining % 3600) // 60)
                    remaining_sec = int(estimated_remaining % 60)
                    
                    print(f"{total_simulated/ms:.1f} ms ({progress_pct:.0f}%) simulated in {elapsed_min}m {elapsed_sec}s, estimated {remaining_hr}h {remaining_min}m {remaining_sec}s remaining.")
                else:
                    print(f"{total_simulated/ms:.1f} ms ({progress_pct:.0f}%) simulated in {elapsed_min}m {elapsed_sec}s, estimated time calculating...")
                
                sys.stdout.flush()
                last_reported_time = total_simulated
                
                # Periodic garbage collection (every 1000ms of simulation time)
                if int(total_simulated/ms) % 1000 == 0 and total_simulated > 0*ms:
                    gc.collect()

        results = {
            'spike_monitors': spike_monitors,
            'voltage_monitors': voltage_monitors,
            'poisson_monitors': poisson_monitors,
            'neuron_groups': neuron_groups,
            'firing_rates': {},
            'synapses': synapse_map,
            'connections': connections
        }
        
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