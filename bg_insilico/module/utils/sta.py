# -*- coding: utf-8 -*-
# Copyright (c) 2025. All rights reserved.
# Author: keun (Jieun Kim)

from brian2 import *
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from brian2 import ms

try:
    matplotlib.use('TkAgg')
except:
    try:
        matplotlib.use('Qt5Agg')
    except:
        matplotlib.use('Agg')

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

# Calculate ISI statistics
def calculate_isi_from_monitor(spike_monitor, neuron_name):
    spike_times, spike_indices = get_monitor_spikes(spike_monitor)
    
    if len(spike_times) == 0:
        return {
            'neuron_name': neuron_name,
            'total_spikes': 0,
            'mean_isi': 0,
            'std_isi': 0,
            'cv_isi': 0,
            'min_isi': 0,
            'max_isi': 0,
            'isi_list': [],
            'isi_by_neuron': {}
        }
    
    isi_by_neuron = {}
    all_isi = []
    
    for neuron_idx in np.unique(spike_indices):
        neuron_spikes = spike_times[spike_indices == neuron_idx]
        if len(neuron_spikes) > 1:
            isi = np.diff(neuron_spikes)
            isi_ms = isi / ms 
            isi_by_neuron[neuron_idx] = isi_ms
            all_isi.extend(isi_ms)
    
    if not all_isi:
        return {
            'neuron_name': neuron_name,
            'total_spikes': len(spike_times),
            'mean_isi': 0,
            'std_isi': 0,
            'cv_isi': 0,
            'min_isi': 0,
            'max_isi': 0,
            'isi_list': [],
            'isi_by_neuron': isi_by_neuron
        }
    
    mean_isi = np.mean(all_isi)
    std_isi = np.std(all_isi)
    cv_isi = std_isi / mean_isi if mean_isi > 0 else 0
    
    return {
        'neuron_name': neuron_name,
        'total_spikes': len(spike_times),
        'mean_isi': mean_isi,
        'std_isi': std_isi,
        'cv_isi': cv_isi,
        'min_isi': np.min(all_isi),
        'max_isi': np.max(all_isi),
        'isi_list': all_isi,
        'isi_by_neuron': isi_by_neuron
    }

# Analyze spike patterns
def analyze_spike_patterns(spike_monitors, neuron_names=None):
    if neuron_names is None:
        neuron_names = list(spike_monitors.keys())
    
    pattern_results = {}
    
    for neuron_name in neuron_names:
        if neuron_name in spike_monitors:
            monitor = spike_monitors[neuron_name]
            isi_result = calculate_isi_from_monitor(monitor, neuron_name)
            pattern_results[neuron_name] = isi_result
    
    return pattern_results

# Compare spike patterns
def compare_spike_patterns(pattern_results1, pattern_results2, label1="Condition 1", label2="Condition 2"):
    comparison_results = {}
    common_neurons = set(pattern_results1.keys()) & set(pattern_results2.keys())
    
    for neuron_name in common_neurons:
        result1 = pattern_results1[neuron_name]
        result2 = pattern_results2[neuron_name]
        
        comparison_results[neuron_name] = {
            label1: result1,
            label2: result2
        }
        
        if result1['mean_isi'] > 0 and result2['mean_isi'] > 0:
            isi_change = ((result2['mean_isi'] - result1['mean_isi']) / result1['mean_isi']) * 100
            cv_change = result2['cv_isi'] - result1['cv_isi']
            
            if result1['total_spikes'] > 0:
                spike_change = ((result2['total_spikes'] - result1['total_spikes']) / result1['total_spikes']) * 100
            else:
                spike_change = 0
            
            comparison_results[neuron_name]['changes'] = {
                'isi_change_percent': isi_change,
                'cv_change': cv_change,
                'spike_change_percent': spike_change
            }
    
    return comparison_results