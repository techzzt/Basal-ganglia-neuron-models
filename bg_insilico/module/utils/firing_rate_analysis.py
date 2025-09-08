# -*- coding: utf-8 -*-
# Copyright (c) 2025. All rights reserved.
# Author: keun (Jieun Kim)

import numpy as np
from brian2 import ms, second

def get_monitor_spikes(monitor):
    try:
        if hasattr(monitor, 't') and hasattr(monitor, 'i') and len(monitor.t) > 0:
            return monitor.t, monitor.i
        else:
            return np.array([]) * ms, np.array([])
    except:
        return np.array([]) * ms, np.array([])

def calculate_and_print_firing_rates(spike_monitors, start_time=2000*ms, end_time=10000*ms, display_names=None):
    firing_rates = {}
    
    for group_name, monitor in spike_monitors.items():
        if hasattr(monitor, 't') and hasattr(monitor, 'i') and len(monitor.t) > 0:
            t_ms = monitor.t / ms
            mask = (t_ms >= start_time/ms) & (t_ms <= end_time/ms)
            spikes_in_window = np.sum(mask)
            duration_sec = (end_time - start_time) / 1000.0
            N = monitor.source.N
            firing_rate = spikes_in_window / (N * duration_sec)
            firing_rates[group_name] = firing_rate
            display_name = display_names.get(group_name, group_name) if display_names else group_name
            print(f"{display_name:<20}: {firing_rate:>8.2f} Hz")
        else:
            firing_rates[group_name] = 0.0
            display_name = display_names.get(group_name, group_name) if display_names else group_name
            print(f"{display_name:<20}: {0.0:>8.2f} Hz")
    
    return firing_rates

def analyze_input_rates_and_spike_counts(spike_monitors, ext_inputs, neuron_configs, stimulus_config, analysis_start_time, analysis_end_time):
    for group_name, monitor in spike_monitors.items():
        if hasattr(monitor, 't') and hasattr(monitor, 'i') and len(monitor.t) > 0:
            spike_times, spike_indices = get_monitor_spikes(monitor)
            spike_times_ms = spike_times / ms
            mask = (spike_times_ms >= analysis_start_time/ms) & (spike_times_ms <= analysis_end_time/ms)
            spike_count = np.sum(mask)
            total_neurons = monitor.source.N
            print(f"{group_name}: {spike_count:,} spikes from {total_neurons:,} neurons")
        else:
            print(f"{group_name}: 0 spikes")

def analyze_firing_rates_by_stimulus_periods(spike_monitors, stimulus_config, analysis_start_time=2000*ms, plot_order=None, display_names=None):
    if not stimulus_config.get('enabled', False):
        return

    stim_start = stimulus_config.get('start_time', 10000) * ms
    stim_duration = stimulus_config.get('duration', 1000) * ms
    stim_end = stim_start + stim_duration
    pre_stim_start = analysis_start_time
    pre_stim_end = stim_start
    post_stim_start = stim_end

    if plot_order:
        monitors_to_analyze = {name: spike_monitors[name] for name in plot_order if name in spike_monitors}
    else:
        monitors_to_analyze = spike_monitors
    
    for name, monitor in monitors_to_analyze.items():
        display_name = display_names.get(name, name) if display_names else name
        spike_times, spike_indices = get_monitor_spikes(monitor)
        total_neurons = monitor.source.N
        spike_times_ms = spike_times / ms
        
        pre_mask = (spike_times >= pre_stim_start) & (spike_times < pre_stim_end)
        pre_spikes = np.sum(pre_mask)
        pre_duration_sec = (pre_stim_end - pre_stim_start) / second
        pre_rate = pre_spikes / (total_neurons * pre_duration_sec) if pre_duration_sec > 0 else 0
        
        stim_mask = (spike_times >= stim_start) & (spike_times < stim_end)
        stim_spikes = np.sum(stim_mask)
        stim_duration_sec = stim_duration / second
        stim_rate = stim_spikes / (total_neurons * stim_duration_sec) if stim_duration_sec > 0 else 0
        
        post_end = post_stim_start + (pre_stim_end - pre_stim_start)
        post_mask = (spike_times >= post_stim_start) & (spike_times < post_end)
        post_spikes = np.sum(post_mask)
        post_duration_sec = (post_end - post_stim_start) / second
        post_rate = post_spikes / (total_neurons * post_duration_sec) if post_duration_sec > 0 else 0
        
        stim_change = ((stim_rate - pre_rate) / pre_rate * 100) if pre_rate > 0 else 0
        post_change = ((post_rate - pre_rate) / pre_rate * 100) if pre_rate > 0 else 0
        
        print(f"{display_name}: Pre={pre_rate:.3f}Hz, During={stim_rate:.3f}Hz, Post={post_rate:.3f}Hz, Effect={stim_change:+.1f}%, Recovery={post_change:+.1f}%")

def compare_firing_rates_between_conditions(normal_monitors, pd_monitors, start_time=2000*ms, end_time=10000*ms, display_names=None):
    normal_rates = {}
    pd_rates = {}
    
    for group_name, monitor in normal_monitors.items():
        if hasattr(monitor, 't') and hasattr(monitor, 'i') and len(monitor.t) > 0:
            t_ms = monitor.t / ms
            mask = (t_ms >= start_time/ms) & (t_ms <= end_time/ms)
            spikes_in_window = np.sum(mask)
            duration_sec = (end_time - start_time) / 1000.0
            N = monitor.source.N
            firing_rate = spikes_in_window / (N * duration_sec)
            normal_rates[group_name] = firing_rate
        else:
            normal_rates[group_name] = 0.0
    
    for group_name, monitor in pd_monitors.items():
        if hasattr(monitor, 't') and hasattr(monitor, 'i') and len(monitor.t) > 0:
            t_ms = monitor.t / ms
            mask = (t_ms >= start_time/ms) & (t_ms <= end_time/ms)
            spikes_in_window = np.sum(mask)
            duration_sec = (end_time - start_time) / 1000.0
            N = monitor.source.N
            firing_rate = spikes_in_window / (N * duration_sec)
            pd_rates[group_name] = firing_rate
        else:
            pd_rates[group_name] = 0.0
    
    for group_name in normal_rates.keys():
        if group_name in pd_rates:
            normal_rate = normal_rates[group_name]
            pd_rate = pd_rates[group_name]
            
            if normal_rate > 0:
                change = ((pd_rate - normal_rate) / normal_rate * 100)
            else:
                change = 0 if pd_rate == 0 else float('inf')
            
            display_name = display_names.get(group_name, group_name) if display_names else group_name
            print(f"{display_name}: Normal={normal_rate:.2f}Hz, PD={pd_rate:.2f}Hz, Change={change:+.1f}%")
    
    return normal_rates, pd_rates
