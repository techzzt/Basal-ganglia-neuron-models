# -*- coding: utf-8 -*-
# Copyright (c) 2025. All rights reserved.
# Author: keun (Jieun Kim)
# Description: Poisson spike train generator

from brian2 import *
import numpy as np

# Generate non-overlapping spikes
def _generate_non_overlapping_spike_trains(N, rate_total_hz_array, dt_ms, duration_ms, simulation_dt_ms=0.1):
    steps = int(np.ceil(duration_ms / dt_ms))
    if len(rate_total_hz_array) != steps:
        if len(rate_total_hz_array) < steps:
            last = rate_total_hz_array[-1] if len(rate_total_hz_array) > 0 else 0.0
            rate_total_hz_array = np.pad(rate_total_hz_array, (0, steps - len(rate_total_hz_array)), constant_values=last)
        else:
            rate_total_hz_array = rate_total_hz_array[:steps]

    indices = []
    times_ms = []
    
    rate_total_per_ms = rate_total_hz_array / 1000.0

    rng = np.random.default_rng()
    for step in range(steps):
        bin_start = step * dt_ms
        lam = rate_total_per_ms[step] * dt_ms
        k = rng.poisson(lam)
        if k <= 0:
            continue

        chosen = rng.integers(0, N, size=k)
        t_uniform = bin_start + rng.random(size=k) * dt_ms
        
        seen_neurons = set()
        for idx, t_val in zip(chosen, t_uniform):
            if idx not in seen_neurons:
                indices.append(idx)
                times_ms.append(t_val)
                seen_neurons.add(idx)

    if len(indices) == 0:
        return np.array([], dtype=int), np.array([], dtype=float)

    indices = np.asarray(indices, dtype=int)
    times_ms = np.asarray(times_ms, dtype=float)
    
    sort_idx = np.argsort(times_ms)
    indices = indices[sort_idx]
    times_ms = times_ms[sort_idx]
    
    last_spike_time = {}
    final_indices = []
    final_times = []
    
    for idx, t in zip(indices, times_ms):
        if idx not in last_spike_time or (t - last_spike_time[idx]) >= simulation_dt_ms:
            final_indices.append(idx)
            final_times.append(t)
            last_spike_time[idx] = t
    
    return np.asarray(final_indices, dtype=int), np.asarray(final_times, dtype=float)

# Create Poisson inputs
def create_poisson_inputs(neuron_groups, external_inputs, scaled_neuron_counts, neuron_configs=None, amplitude_oscillations=None, stimulus_config=None, simulation_params=None):
    poisson_groups = {}
    
    total_duration = 10000 
    if simulation_params:
        total_duration = simulation_params.get('duration', 10000)
    
    if stimulus_config and stimulus_config.get('enabled', False):
        stim_start = stimulus_config.get('start_time', 10000)
        stim_duration = stimulus_config.get('duration', 1000) 
        stim_rates = stimulus_config.get('rates', {})
        dt_array = stimulus_config.get('dt_array', 1)
        use_stimulus = True
    else:
        use_stimulus = False
        dt_array = (stimulus_config or {}).get('dt_array', 1)

    try:
        dt_candidate = float(dt_array)
    except Exception:
        dt_candidate = float(simulation_params.get('dt', 1) if simulation_params else 1)
    if not np.isfinite(dt_candidate) or dt_candidate <= 0:
        dt_candidate = float(simulation_params.get('dt', 1) if simulation_params else 1)
    if not np.isfinite(dt_candidate) or dt_candidate <= 0:
        dt_candidate = 1.0
    dt_bins = max(0.05, min(dt_candidate, 0.05))  
    
    for target, rate_expr in external_inputs.items():
        if target not in neuron_groups:
            continue
        
        try:
            N_post = int(neuron_groups[target].N)

            external_conf = None
            if neuron_configs:
                for nc in neuron_configs:
                    if nc.get('neuron_type') == 'poisson' and 'target_rates' in nc:
                        cfg_target, _ = list(nc['target_rates'].items())[0]
                        if cfg_target == target:
                            external_conf = nc
                            break

            N_pre = int(external_conf['N']) if external_conf and 'N' in external_conf else N_post

            if isinstance(rate_expr, str) and '*Hz' in rate_expr:
                rate_per_stream = float((eval(rate_expr))/Hz)
            elif isinstance(rate_expr, str):
                rate_per_stream = float((eval(rate_expr) * Hz)/Hz)
            else:
                rate_per_stream = float((rate_expr * Hz)/Hz)

            total_population_rate = rate_per_stream * N_pre

            array_duration = max(1, int(total_duration))
            time_points = np.arange(0, array_duration, dt_bins, dtype=float)

            rates_total_hz = np.full(len(time_points), total_population_rate, dtype=float)

            if use_stimulus and target in stim_rates:
                stim_rate_total = float(stim_rates[target]) * N_pre
                stim_start_idx = int(stim_start / dt_bins)
                stim_end_idx = int((stim_start + stim_duration) / dt_bins)
                if stim_start_idx < len(rates_total_hz):
                    end_idx = min(stim_end_idx, len(rates_total_hz))
                    rates_total_hz[stim_start_idx:end_idx] = stim_rate_total

            simulation_dt_ms = float(simulation_params.get('dt', 0.1)) if simulation_params else 0.1
            idx, t_ms = _generate_non_overlapping_spike_trains(
                N=N_pre, rate_total_hz_array=rates_total_hz, dt_ms=dt_bins, duration_ms=array_duration, simulation_dt_ms=simulation_dt_ms
            )
            group = SpikeGeneratorGroup(N_pre, idx, t_ms * ms)
            
            external_name = None
            if external_conf:
                external_name = external_conf.get('name')
            else:
                for neuron_config in neuron_configs:
                    if neuron_config.get('neuron_type') == 'poisson':
                        if 'target_rates' in neuron_config:
                            config_target, _ = list(neuron_config['target_rates'].items())[0]
                            if config_target == target:
                                external_name = neuron_config['name']
                                break
            
            if external_name:
                poisson_groups[external_name] = group
            else:
                poisson_groups[target] = group

        except Exception:
            pass
    
    return poisson_groups, []