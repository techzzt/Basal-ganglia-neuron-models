# -*- coding: utf-8 -*-
# Copyright (c) 2025. All rights reserved.
# Author: keun (Jieun Kim)

import matplotlib
import matplotlib.pyplot as plt
import os
import numpy as np 
from brian2 import *

# Set matplotlib backend
backend = os.environ.get('MPLBACKEND', None)
if backend:
    matplotlib.use(backend)
else:
    try:
        matplotlib.use('TkAgg')
    except:
        try:
            matplotlib.use('Qt5Agg')
        except:
            matplotlib.use('Agg')

plt.ion()  

# Apply Gaussian smoothing to 1D data
def gaussian_smooth(data, sigma):
    if sigma <= 0:
        return data

    kernel_size = int(6 * sigma)
    if kernel_size % 2 == 0:
        kernel_size += 1
    x = np.arange(kernel_size) - kernel_size // 2
    kernel = np.exp(-0.5 * (x / sigma) ** 2)
    kernel = kernel / np.sum(kernel)
    
    padded_data = np.pad(data, kernel_size//2, mode='edge')
    smoothed = np.convolve(padded_data, kernel, mode='valid')
    return smoothed

# Extract spike times and indices from monitor
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

# Create raster plot with improved visualization
def plot_improved_overall_raster(spike_monitors, sample_size=12, plot_order=None, 
                                start_time=0*ms, end_time=1000*ms, display_names=None, 
                                stimulus_periods=None, save_plot=True,
                               visual_thinning=True, max_points_per_group=4000, 
                               max_spikes_per_neuron=150):
    try:
        if plot_order:
            spike_monitors = {name: spike_monitors[name] for name in plot_order if name in spike_monitors}

        if not spike_monitors:
            return
        
        n_plots = len(spike_monitors)
        fig, axes = plt.subplots(n_plots, 1, figsize=(12, 1.5 * n_plots), sharex=True)
        if n_plots == 1:
            axes = [axes]
          
        for plot_idx, (name, monitor) in enumerate(spike_monitors.items()):
            spike_times, spike_indices = get_monitor_spikes(monitor)
            
            if len(spike_indices) == 0:
                continue
            
            total_neurons = monitor.source.N
            sample_size_local = min(sample_size, total_neurons)

            chosen_neurons = np.random.choice(total_neurons, size=sample_size_local, replace=False)
            chosen_neurons = np.sort(chosen_neurons)

            time_mask = (spike_times >= start_time) & (spike_times <= end_time)
            neuron_mask = np.isin(spike_indices, chosen_neurons)
            combined_mask = time_mask & neuron_mask

            display_t = spike_times[combined_mask]
            display_i = spike_indices[combined_mask]

            neuron_mapping = {original: new for new, original in enumerate(chosen_neurons)}
            remapped_i = [neuron_mapping[original] for original in display_i]

            display_name = display_names.get(name, name) if display_names else name
            plotted_t = display_t
            plotted_i = np.array(remapped_i)

            if visual_thinning and len(display_t) > 0:
                idx_all = np.arange(len(display_t))
                keep_mask = np.zeros(len(display_t), dtype=bool)
                unique_ids = np.unique(plotted_i)

                for uid in unique_ids:
                    uid_mask = (plotted_i == uid)
                    uid_idx = idx_all[uid_mask]

                    if len(uid_idx) <= max_spikes_per_neuron:
                        keep_mask[uid_idx] = True
                    else:
                        keep_idx = np.random.choice(uid_idx, size=max_spikes_per_neuron, replace=False)
                        keep_mask[keep_idx] = True
                kept_idx = np.where(keep_mask)[0]

                if len(kept_idx) > max_points_per_group:
                    kept_idx = np.random.choice(kept_idx, size=max_points_per_group, replace=False)
                plotted_t = display_t[kept_idx]
                plotted_i = plotted_i[kept_idx]

            if len(plotted_t) > 0:
                axes[plot_idx].scatter(plotted_t / ms, plotted_i, s=2.5, alpha=0.8, color='#4682B4')
            axes[plot_idx].set_title(f'{display_name}', fontsize=14, pad=2)

            if len(chosen_neurons) > 0:
                axes[plot_idx].set_ylim(-0.5, len(chosen_neurons) - 0.5)
            axes[plot_idx].set_yticks([])
            axes[plot_idx].set_yticklabels([])
            axes[plot_idx].set_ylabel('')
            axes[plot_idx].set_xlim(int(start_time/ms), int(end_time/ms))
            
            for j in range(0, len(chosen_neurons), max(1, len(chosen_neurons)//10)):
                axes[plot_idx].axhline(y=j-0.5, color='gray', alpha=0.2, linewidth=0.3)
            
            if stimulus_periods:
                for period_idx, (stim_start, stim_end) in enumerate(stimulus_periods):
                    if stim_start >= start_time and stim_end <= end_time:
                        axes[plot_idx].axvspan(stim_start/ms, stim_end/ms, alpha=0.2, color='red')

        axes[-1].set_xlabel('Time (ms)', fontsize=12)
        plt.tight_layout(pad=0.5) 
        
        if save_plot:
            filename = 'improved_raster_plot.png'
            plt.savefig(filename, dpi=300, bbox_inches='tight')
        
        try:
            if os.environ.get('MPLBACKEND') == 'Agg':
                plt.close()  
            else:
                plt.show(block=True) 
        except:
            pass

    except:
        pass

# Plot FFT spectra for firing rates
def plot_firing_rate_fft_multi_page(
    spike_monitors, 
    neuron_indices=None, 
    start_time=0*ms, 
    end_time=10000*ms, 
    bin_size=10*ms, 
    show_mean=True, 
    max_freq=60, 
    title='Firing Rate FFT Spectra',
    display_names=None,
    comparison_monitors=None, 
    comparison_label='PD'    
):
    try:
        if not spike_monitors:
            return
            
        neuron_groups = list(spike_monitors.keys())
        n_groups = len(neuron_groups)
        
        if n_groups == 0:
            return
            
        cols = 4
        rows = 2
        fig, axes = plt.subplots(rows, cols, figsize=(12, 7))
        
        if n_groups == 1:
            axes = [axes]
        elif rows == 1:
            axes = axes.reshape(1, -1)
        
        colors = ['#E74C3C', '#8E44AD', '#3498DB', '#27AE60', '#F39C12', '#34495E', '#E67E22']
        
        for idx, group_name in enumerate(neuron_groups):
            row = idx // cols
            col = idx % cols
            ax = axes[row, col] if rows > 1 else axes[col]
            
            spike_monitor = spike_monitors[group_name]
            spike_times = spike_monitor.t / ms
            spike_indices = spike_monitor.i
            N = spike_monitor.source.N

            if neuron_indices is None:
                neuron_indices = range(N)
            
            time_bins = np.arange(start_time/ms, end_time/ms + bin_size/ms, bin_size/ms)
            time_centers = time_bins[:-1] + bin_size/ms/2

            all_spectra = []
            all_freqs = None
            n_valid = 0
            total_neurons = len(neuron_indices)
            
            has_comparison = (comparison_monitors is not None and 
                            group_name in comparison_monitors and 
                            len(comparison_monitors[group_name].t) > 0)
            
            for neuron_idx in neuron_indices:
                neuron_spikes = spike_times[spike_indices == neuron_idx]
                if len(neuron_spikes) == 0:
                    continue
                
                counts, _ = np.histogram(neuron_spikes, bins=time_bins)
                firing_rate = counts / (bin_size/ms/1000.0)
                if np.all(firing_rate == 0):
                    continue

                fft_result = np.fft.fft(firing_rate)
                freqs = np.fft.fftfreq(len(firing_rate), d=(bin_size/ms/1000.0))
                pos_mask = freqs >= 0

                if all_freqs is None:
                    all_freqs = freqs[pos_mask]
                
                power_spectrum = np.abs(fft_result[pos_mask])**2
                normalized_spectrum = power_spectrum / len(firing_rate)
                normalized_spectrum[0] = 0
                
                all_spectra.append(normalized_spectrum)
                n_valid += 1

                if len(neuron_indices) <= 5:
                    ax.plot(freqs[pos_mask], normalized_spectrum, alpha=0.4, linewidth=0.8, color=colors[idx % len(colors)])

            all_spectra = np.array(all_spectra)
            if show_mean and len(all_spectra) > 0:
                mean_spectrum = np.mean(all_spectra, axis=0)
                ax.plot(all_freqs, mean_spectrum, color='#808080', linestyle='-', linewidth=2, label='Normal')

            if has_comparison:
                comp_monitor = comparison_monitors[group_name]
                comp_spike_times = comp_monitor.t / ms
                comp_spike_indices = comp_monitor.i
                
                comp_spectra = []
                for neuron_idx in neuron_indices:
                    neuron_spikes = comp_spike_times[comp_spike_indices == neuron_idx]
                    if len(neuron_spikes) == 0:
                        continue
                    
                    counts, _ = np.histogram(neuron_spikes, bins=time_bins)
                    firing_rate = counts / (bin_size/ms/1000.0)
                    if np.all(firing_rate == 0):
                        continue

                    fft_result = np.fft.fft(firing_rate)
                    freqs = np.fft.fftfreq(len(firing_rate), d=(bin_size/ms/1000.0))
                    pos_mask = freqs >= 0

                    power_spectrum = np.abs(fft_result[pos_mask])**2
                    normalized_spectrum = power_spectrum / len(firing_rate)
                    normalized_spectrum[0] = 0
                    
                    comp_spectra.append(normalized_spectrum)

                if len(comp_spectra) > 0:
                    comp_spectra = np.array(comp_spectra)
                    comp_mean_spectrum = np.mean(comp_spectra, axis=0)
                    ax.plot(all_freqs, comp_mean_spectrum, color='#C00000', linestyle='--', linewidth=2, label=comparison_label)
            
            display_name = display_names.get(group_name, group_name) if display_names else group_name
            ax.set_title(f'{display_name}', fontweight='bold', fontsize=10)
            ax.set_xlim(0, max_freq)
            ax.set_ylabel('Power', fontsize=8)
            
            if has_comparison:
                ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)
                    
        for idx in range(n_groups, rows * cols):
            if rows > 1:
                row = idx // cols
                col = idx % cols
                axes[row, col].set_visible(False)
            elif cols > 1:
                axes[idx].set_visible(False)
        
        for ax in axes.flat if hasattr(axes, 'flat') else axes:
            ax.set_xlabel('Frequency (Hz)', fontsize=10)
        
        plt.tight_layout()
        plt.show(block=True)
        
    except:
        pass