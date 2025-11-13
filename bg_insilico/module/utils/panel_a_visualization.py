# -*- coding: utf-8 -*-
# Copyright (c) 2025. All rights reserved.
# Author: keun (Jieun Kim)

import matplotlib
import matplotlib.pyplot as plt
import os
import numpy as np
from brian2 import *
from module.utils.data_storage import load_simulation_results
        
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

# Gaussian kernel for smoothing
def _gaussian_kernel(sigma_ms, bin_ms):
    kernel_size = int(6 * sigma_ms / bin_ms)
    if kernel_size % 2 == 0:
        kernel_size += 1
    x = np.arange(kernel_size) - kernel_size // 2
    kernel = np.exp(-0.5 * (x * bin_ms / sigma_ms) ** 2)
    return kernel / np.sum(kernel)

# Bandpass filter using FFT
def _bandpass_fft(signal, fs_hz, f_lo, f_hi):
    try:
        from scipy import signal as scipy_signal
        sos = scipy_signal.butter(4, [f_lo, f_hi], btype='band', fs=fs_hz, output='sos')
        return scipy_signal.sosfilt(sos, signal)
    except ImportError:

        fft_result = np.fft.fft(signal)
        freqs = np.fft.fftfreq(len(signal), d=1.0/fs_hz)
        mask = (freqs >= f_lo) & (freqs <= f_hi)
        fft_result[~mask] = 0
        return np.real(np.fft.ifft(fft_result))

def plot_panel_a_style(spike_monitors=None, pkl_file=None,
                       start_time=2000*ms, end_time=10000*ms,
                       group_sets=None,
                       display_names=None,
                       lfp_bin_ms=2.0,
                       rate_bin_ms=10.0,
                       lfp_sigma_ms=5.0,
                       beta_band=(13.0, 30.0),
                       n_raster_neurons=150,
                       prefix='panelA',
                       save_plot=True,
                       save_eps=False,
                       normal_pkl=None,
                       pd_pkl=None):

    try:

        if spike_monitors is None and pkl_file:
            print(f"Loading data from {pkl_file}...")
            data = load_simulation_results(pkl_file)
            if data is None:
                print(f"Error: Failed to load data from {pkl_file}")
                return

            spike_monitors = data['spike_monitors']
            voltage_monitors = data.get('voltage_monitors', {})

        elif spike_monitors is None:
            print("Error: Either spike_monitors or pkl_file must be provided")
            return

        else:
            voltage_monitors = {}
        
        if group_sets is None:
            available_groups = list(spike_monitors.keys())
            group_sets = [([name], name) for name in available_groups]
        
        start_ms = float(start_time / ms)
        end_ms = float(end_time / ms)
        fs_hz = 1000.0 / lfp_bin_ms
        
        # Process each group set
        for group_names, disp_name in group_sets:
            all_t, all_i, offsets, total_N = [], [], [], 0
            
            for g in group_names:
                if g not in spike_monitors:
                    continue
                
                mon = spike_monitors[g]
                t, i = get_monitor_spikes(mon)
                
                if hasattr(t, '__iter__') and len(t) > 0:
                    t_ms = np.array([float(tt/ms) for tt in t])
                else:
                    t_ms = np.array([])
                
                i_arr = np.array(i, dtype=int)
                N = int(mon.source.N)
                
                time_mask = (t_ms >= start_ms) & (t_ms <= end_ms)
                t_sel = t_ms[time_mask]
                i_sel = i_arr[time_mask] + total_N  
                
                all_t.append(t_sel)
                all_i.append(i_sel)
                offsets.append(total_N)
                total_N += N
            
            if total_N == 0 or len(all_t) == 0:
                print(f"Warning: No spikes found for {disp_name}")
                continue
            
            t_all = np.concatenate(all_t) if len(all_t) > 0 else np.array([])
            i_all = np.concatenate(all_i) if len(all_i) > 0 else np.array([], dtype=int)
            
            if t_all.size == 0:
                print(f"Warning: No spikes in time window for {disp_name}")
                continue
            
            lfp_tau_ms = 20.0  
            lfp_contribution_per_spike_uv = 0.5  
            
            # Create time bins for LFP
            n_bins_lfp = max(10, int(np.ceil((end_ms - start_ms) / lfp_bin_ms)))
            edges_lfp = np.linspace(start_ms, end_ms, n_bins_lfp + 1)
            time_bins_ms = edges_lfp
            
            spike_counts, _ = np.histogram(t_all, bins=time_bins_ms)
            
            max_kernel_time = 5 * lfp_tau_ms  
            kernel_times = np.arange(0, max_kernel_time + lfp_bin_ms, lfp_bin_ms)
            kernel = np.exp(-kernel_times / lfp_tau_ms)
            
            lfp_wideband = np.convolve(spike_counts, kernel, mode='same')
            lfp_wideband_uv = lfp_wideband * lfp_contribution_per_spike_uv
            kernel_smooth = _gaussian_kernel(lfp_sigma_ms, lfp_bin_ms)
            lfp_smoothed = np.convolve(lfp_wideband_uv, kernel_smooth, mode='same')
            
            # Bandpass filter to beta band
            beta_muv = _bandpass_fft(lfp_smoothed, fs_hz=fs_hz, f_lo=beta_band[0], f_hi=beta_band[1])
            
            # Binning for population firing rate
            n_bins_rate = max(10, int(np.ceil((end_ms - start_ms) / rate_bin_ms)))
            edges_rate = np.linspace(start_ms, end_ms, n_bins_rate + 1)
            counts_rate, _ = np.histogram(t_all, bins=edges_rate)
            pop_rate = counts_rate / (rate_bin_ms / 1000.0)  

            unique_neurons = np.unique(i_all)
            if unique_neurons.size == 0:
                continue

            part_size = total_N // 3
            bounds = [(0, part_size), (part_size, 2 * part_size), (2 * part_size, total_N)]
            per_part = max(1, n_raster_neurons // 3)
            sampled_parts = []
            
            for lo, hi in bounds:
                candidates = unique_neurons[(unique_neurons >= lo) & (unique_neurons < hi)]
                if candidates.size > 0:
                    k = min(per_part, candidates.size)
                    sampled_parts.append(np.random.choice(candidates, size=k, replace=False))
            
            sampled = np.concatenate(sampled_parts) if len(sampled_parts) > 0 else np.array([], dtype=int)

            if sampled.size < min(n_raster_neurons, unique_neurons.size):
                remaining = np.setdiff1d(unique_neurons, sampled, assume_unique=False)
                need = min(n_raster_neurons, unique_neurons.size) - sampled.size
                if need > 0 and remaining.size > 0:
                    sampled = np.concatenate([sampled, np.random.choice(remaining, size=min(need, remaining.size), replace=False)])
            
            raster_mask = np.isin(i_all, sampled)
            t_raster = t_all[raster_mask]
            i_raster = i_all[raster_mask]
            
            neuron_to_row = {n: idx for idx, n in enumerate(np.sort(sampled))}
            r_rows = np.array([neuron_to_row[n] for n in i_raster], dtype=int)
            n_sample = len(sampled)
            
            t_wide = (edges_lfp[:-1] + edges_lfp[1:]) * 0.5
            t_rate = (edges_rate[:-1] + edges_rate[1:]) * 0.5
            
            fig, axes = plt.subplots(3, 1, figsize=(14, 8), sharex=True)
            
            fig.patch.set_facecolor('none')
            for ax in axes:
                ax.set_facecolor('none')
            
            axes[0].plot(t_wide, beta_muv, color='green', linewidth=1.0, alpha=0.95)
            axes[0].set_ylabel('LFP (μV)', fontsize=10)
            axes[0].grid(False)  
        
            axes[1].bar(t_rate, pop_rate, width=rate_bin_ms * 0.9, color='#1f77b4', alpha=0.9, align='center')
            axes[1].set_ylabel('Rate (spk/s)', fontsize=10)
            axes[1].grid(False)  
            
            if len(t_raster) > 0:
                axes[2].scatter(t_raster, r_rows, s=2, color='#d62728', alpha=0.8, linewidths=0, edgecolors='none')
            axes[2].set_ylabel('Sample neurons', fontsize=10)
            axes[2].set_xlabel('Time (ms)', fontsize=12)
            axes[2].set_ylim(-1, n_sample + 1)
            axes[2].grid(False) 
            
            axes[0].set_title(f"{disp_name} — LFP, population rate, raster")
            axes[-1].set_xlim(start_ms, end_ms)
            plt.tight_layout()
            
            safe_name = disp_name.replace(' ', '_').replace('/', '_')
            png = f"{prefix}_panelA_{safe_name}.png"
            eps = f"{prefix}_panelA_{safe_name}.eps"
            
            plt.savefig(png, dpi=300, bbox_inches='tight', facecolor='none', transparent=True)
            print(f"LFP + Raster plot saved to {png}")
            
            plt.savefig(eps, format='eps', dpi=300, bbox_inches='tight', facecolor='none', transparent=True)
            print(f"LFP + Raster plot saved to {eps}")

            try:
                zoom_lo, zoom_hi = 4500.0, 5000.0

                for ax in axes:
                    ax.set_xlim(zoom_lo, zoom_hi)
                png_zoom = f"{prefix}_panelA_{safe_name}_zoom_{int(zoom_lo)}_{int(zoom_hi)}.png"
                eps_zoom = f"{prefix}_panelA_{safe_name}_zoom_{int(zoom_lo)}_{int(zoom_hi)}.eps"
                plt.savefig(png_zoom, dpi=300, bbox_inches='tight', facecolor='none', transparent=True)
                print(f"Zoomed plot ({{zoom_lo}}-{{zoom_hi}} ms) saved to {png_zoom}")
                plt.savefig(eps_zoom, format='eps', dpi=300, bbox_inches='tight', facecolor='none', transparent=True)
                print(f"Zoomed plot ({{zoom_lo}}-{{zoom_hi}} ms) saved to {eps_zoom}")
            except Exception as _:
                pass

        try:
            if isinstance(voltage_monitors, dict) and 'SNr' in voltage_monitors:
                vm = voltage_monitors['SNr']
                t_vm = np.array([float(tt/ms) for tt in vm.t]) if hasattr(vm, 't') else None
                v_arr = np.array(vm.v / mV) if hasattr(vm, 'v') else None
                if t_vm is not None and v_arr is not None and t_vm.size > 0 and v_arr.size > 0:
                    if v_arr.ndim == 1:
                        v_arr = v_arr[np.newaxis, :]
                    if v_arr.shape[-1] != t_vm.size:
                        if v_arr.shape[0] == t_vm.size:
                            v_arr = v_arr.T

                    v0 = v_arr[0]
                    fig_mem, ax_mem = plt.subplots(1, 1, figsize=(14, 3.5))
                    fig_mem.patch.set_facecolor('none')
                    ax_mem.set_facecolor('none')
                    ax_mem.plot(t_vm, v0, color='#1f77b4', linewidth=1.0)
                    ax_mem.set_title('SNr — Membrane potential (neuron 0)')
                    ax_mem.set_xlabel('Time (ms)')
                    ax_mem.set_ylabel('V (mV)')
                    ax_mem.grid(False)

                    png_mem = f"{prefix}_panelA_SNr_membrane.png"
                    eps_mem = f"{prefix}_panelA_SNr_membrane.eps"

                    plt.tight_layout()

                    fig_mem.savefig(png_mem, dpi=300, bbox_inches='tight', facecolor='none', transparent=True)
                    fig_mem.savefig(eps_mem, format='eps', dpi=300, bbox_inches='tight', facecolor='none', transparent=True)
                    print(f"SNr membrane plot saved to {png_mem}")
                    print(f"SNr membrane plot saved to {eps_mem}")

                    zoom_lo, zoom_hi = 4500.0, 5000.0
                    ax_mem.set_xlim(zoom_lo, zoom_hi)
                    png_mem_zoom = f"{prefix}_panelA_SNr_membrane_zoom_{int(zoom_lo)}_{int(zoom_hi)}.png"
                    eps_mem_zoom = f"{prefix}_panelA_SNr_membrane_zoom_{int(zoom_lo)}_{int(zoom_hi)}.eps"
                    fig_mem.savefig(png_mem_zoom, dpi=300, bbox_inches='tight', facecolor='none', transparent=True)
                    fig_mem.savefig(eps_mem_zoom, format='eps', dpi=300, bbox_inches='tight', facecolor='none', transparent=True)
                    print(f"SNr membrane zoom plot saved to {png_mem_zoom}")
                    print(f"SNr membrane zoom plot saved to {eps_mem_zoom}")
                    plt.close(fig_mem)

        except Exception as e_mem:
            print(f"Warning: Failed to generate SNr membrane plots: {e_mem}")
            
            try:
                if os.environ.get('MPLBACKEND') == 'Agg':
                    plt.close()
                else:
                    plt.show(block=False)
            except:
                plt.close()
            
    except Exception as e:
        print(f"Error in plot_panel_a_style: {e}")
        import traceback
        traceback.print_exc()
