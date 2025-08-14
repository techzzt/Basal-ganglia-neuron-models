import matplotlib
import matplotlib.pyplot as plt
import os
import numpy as np 
from brian2 import *
import platform
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

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

def gaussian_smooth(data, sigma):
    """Apply Gaussian smoothing to data"""
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

def get_monitor_spikes(monitor):
    """Extract spike times and indices from monitor"""
    try:
        if hasattr(monitor, 't') and hasattr(monitor, 'i') and len(monitor.t) > 0:
            return monitor.t, monitor.i
        elif hasattr(monitor, '_spike_times') and hasattr(monitor, '_spike_indices'):
            return monitor._spike_times, monitor._spike_indices
        else:
            return np.array([]) * ms, np.array([])
    except:
        return np.array([]) * ms, np.array([])

def analyze_firing_rates_by_stimulus_periods(spike_monitors, stimulus_config, analysis_start_time=2000*ms, plot_order=None, display_names=None):
    """Analyze firing rates before, during, and after stimulus periods"""
    if not stimulus_config.get('enabled', False):
        print("Stimulus is disabled.")
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
    
    print("="*60)
    print(f"Stimulus: {stim_start/ms:.0f}-{stim_end/ms:.0f}ms")
    print(f"Pre-stimulus: {pre_stim_start/ms:.0f}-{pre_stim_end/ms:.0f}ms")
    print(f"During-stimulus: {stim_start/ms:.0f}-{stim_end/ms:.0f}ms") 
    print(f"Post-stimulus: after {post_stim_start/ms:.0f}ms")
    
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
        
        print(f"\n[{display_name}] (Total {total_neurons} neurons)")
        print(f"  Pre-stimulus  ({pre_stim_start/ms:.0f}-{pre_stim_end/ms:.0f}ms): {pre_rate:.3f} Hz")
        print(f"  During-stimulus ({stim_start/ms:.0f}-{stim_end/ms:.0f}ms): {stim_rate:.3f} Hz")
        print(f"  Post-stimulus ({post_stim_start/ms:.0f}-{post_end/ms:.0f}ms): {post_rate:.3f} Hz")
        print(f"  Stimulus effect: {stim_change:+.1f}%")
        print(f"  Recovery state: {post_change:+.1f}%")
        
    print("\n" + "="*60)

def plot_improved_overall_raster(spike_monitors, sample_size=12, plot_order=None, 
                                start_time=0*ms, end_time=1000*ms, display_names=None, 
                                stimulus_periods=None, save_plot=True,
                                visual_thinning=True, max_points_per_group=4000, max_spikes_per_neuron=150):
    """Plot raster plots with activity- and time-balanced sampling per group."""
    np.random.seed(2025)
    try:
        print(f"Available spike monitors: {list(spike_monitors.keys())}")
        if plot_order:
            print(f"Plot order: {plot_order}")
            spike_monitors = {name: spike_monitors[name] for name in plot_order if name in spike_monitors}
            print(f"Filtered monitors: {list(spike_monitors.keys())}")

        if not spike_monitors:
            print("No valid neuron groups to plot.")
            return
        
        n_plots = len(spike_monitors)
        fig, axes = plt.subplots(n_plots, 1, figsize=(18, 3.5 * n_plots), sharex=True)
        if n_plots == 1:
            axes = [axes]

        print(f"\nRaster Plot: {start_time/ms:.0f}-{end_time/ms:.0f}ms")
          
        for plot_idx, (name, monitor) in enumerate(spike_monitors.items()):
            spike_times, spike_indices = get_monitor_spikes(monitor)
            
            print(f"Processing {name}: {len(spike_times)} spikes, {len(spike_indices)} indices")
            
            if len(spike_indices) == 0:
                print(f"No spikes recorded for {name}")
                continue
            
            total_neurons = monitor.source.N
            sample_size_local = min(sample_size, total_neurons)

            # Time-balanced + activity-aware sampling
            # 1) filter by time window
            time_mask_all = (spike_times >= start_time) & (spike_times <= end_time)
            t_win = spike_times[time_mask_all]
            i_win = spike_indices[time_mask_all]

            # 2) define windows
            n_windows = 8
            if (end_time - start_time) <= 4*ms:
                n_windows = 1
            bin_edges = np.linspace(start_time/ms, end_time/ms, n_windows+1)
            # window membership for spikes
            if len(t_win) > 0:
                win_idx = np.digitize(t_win/ms, bin_edges, right=False) - 1
                win_idx = np.clip(win_idx, 0, n_windows-1)
            else:
                win_idx = np.array([], dtype=int)

            # 3) per-neuron stats
            # overall spike count per neuron in window
            if len(i_win) > 0:
                max_index = int(np.max(i_win)) if len(i_win) > 0 else -1
                bincount_size = max(total_neurons, max_index+1)
                counts_overall = np.bincount(i_win, minlength=bincount_size)[:total_neurons]
            else:
                counts_overall = np.zeros(total_neurons, dtype=int)

            # windows covered per neuron
            windows_with_spikes = np.zeros(total_neurons, dtype=int)
            if len(i_win) > 0:
                for w in range(n_windows):
                    mask_w = (win_idx == w)
                    if np.any(mask_w):
                        neurons_w = np.unique(i_win[mask_w])
                        windows_with_spikes[neurons_w] += 1

            # z-score of rates (robust)
            counts_nonzero = counts_overall[counts_overall > 0]
            if len(counts_nonzero) >= 5:
                mean_c = np.mean(counts_nonzero)
                std_c = np.std(counts_nonzero) + 1e-9
                rate_z = (counts_overall - mean_c) / std_c
                rate_z = np.clip(rate_z, -2.0, 2.0)
            else:
                rate_z = np.zeros_like(counts_overall, dtype=float)

            # scoring: prioritize temporal coverage, then activity
            alpha, beta = 1.0, 0.3
            scores = alpha * windows_with_spikes.astype(float) + beta * rate_z

            # quiet quota: ensure including some non-spiking neurons to represent baseline
            if len(i_win) > 0:
                spiking_neurons = np.unique(i_win).astype(int)
            else:
                spiking_neurons = np.array([], dtype=int)
            all_neurons = np.arange(total_neurons, dtype=int)
            quiet_candidates = np.setdiff1d(all_neurons, spiking_neurons)
            quiet_fraction = 0.3
            quiet_target = 0
            if sample_size_local > 1:
                quiet_target = min(len(quiet_candidates), max(1, int(round(sample_size_local * quiet_fraction))))
            max_spiking_allowed = max(0, sample_size_local - quiet_target)

            # window-wise quota to ensure time coverage
            quota = max(1, int(np.ceil(sample_size_local / max(1, n_windows))))
            chosen_spiking = set()
            # for each window, pick top-scoring neurons that fired in that window
            for w in range(n_windows):
                if len(i_win) == 0:
                    break
                mask_w = (win_idx == w)
                if not np.any(mask_w):
                    continue
                neurons_w = np.unique(i_win[mask_w])
                if len(neurons_w) == 0:
                    continue
                # sort by score desc
                scores_w = scores[neurons_w]
                order = np.argsort(-scores_w)
                for idx_in_w in order:
                    n_id = int(neurons_w[idx_in_w])
                    if n_id not in chosen_spiking:
                        if len(chosen_spiking) >= max_spiking_allowed:
                            break
                        chosen_spiking.add(n_id)
                        if len(chosen_spiking) >= min(max_spiking_allowed, len(range(total_neurons))):
                            break
                    if len(chosen_spiking) >= len(range(total_neurons)) or len(chosen_spiking) >= max_spiking_allowed:
                        break
                if len(chosen_spiking) >= max_spiking_allowed:
                    break

            # fill remaining with top global scores (including quiet neurons if needed)
            if len(chosen_spiking) < max_spiking_allowed:
                all_indices = np.arange(total_neurons)
                # consider only spiking neurons for global fill
                spiking_mask = np.zeros(total_neurons, dtype=bool)
                spiking_mask[spiking_neurons] = True
                order_global = np.argsort(-scores)
                for idx_global in order_global:
                    n_id = int(all_indices[idx_global])
                    if not spiking_mask[n_id]:
                        continue
                    if n_id not in chosen_spiking:
                        chosen_spiking.add(n_id)
                        if len(chosen_spiking) >= max_spiking_allowed:
                            break

            # choose quiet neurons to reach target and fill to sample size
            remaining_slots = sample_size_local - len(chosen_spiking)
            quiet_pick = min(len(quiet_candidates), max(quiet_target, remaining_slots))
            quiet_selected = []
            if quiet_pick > 0 and len(quiet_candidates) > 0:
                n_strata_q = min(quiet_pick, 5)
                per_stratum = quiet_pick // n_strata_q
                rem_q = quiet_pick % n_strata_q
                for s in range(n_strata_q):
                    start_q = s * len(quiet_candidates) // n_strata_q
                    end_q = (s + 1) * len(quiet_candidates) // n_strata_q
                    seg = quiet_candidates[start_q:end_q]
                    take = per_stratum + (1 if s < rem_q else 0)
                    if take <= 0 or len(seg) == 0:
                        continue
                    if take >= len(seg):
                        chosen_seg = seg
                    else:
                        chosen_seg = np.random.choice(seg, size=take, replace=False)
                    quiet_selected.extend(list(chosen_seg))

            chosen_neurons = sorted(list(chosen_spiking) + quiet_selected)

            time_mask = (spike_times >= start_time) & (spike_times <= end_time)
            neuron_mask = np.isin(spike_indices, chosen_neurons)
            combined_mask = time_mask & neuron_mask

            display_t = spike_times[combined_mask]
            display_i = spike_indices[combined_mask]

            # Create mapping from original neuron indices to display indices (0, 1, 2, ...)
            neuron_mapping = {original: new for new, original in enumerate(chosen_neurons)}
            remapped_i = [neuron_mapping[original] for original in display_i]

            display_name = display_names.get(name, name) if display_names else name

            # Plot each spike at the correct y-position
            # Optional visual thinning to avoid over-plotting artifacts
            plotted_t = display_t
            plotted_i = np.array(remapped_i)
            if visual_thinning and len(display_t) > 0:
                idx_all = np.arange(len(display_t))
                keep_mask = np.zeros(len(display_t), dtype=bool)
                # per-neuron cap
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
                # global cap
                if len(kept_idx) > max_points_per_group:
                    kept_idx = np.random.choice(kept_idx, size=max_points_per_group, replace=False)
                plotted_t = display_t[kept_idx]
                plotted_i = plotted_i[kept_idx]

            if len(plotted_t) > 0:
                axes[plot_idx].scatter(plotted_t / ms, plotted_i, s=2.5, alpha=0.6)
            axes[plot_idx].set_title(f'{display_name} Raster Plot', fontsize=14, pad=15)
            # Hide neuron index labels on Y axis
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
                
                axes[plot_idx].grid(True, alpha=0.08, axis='x')
            
            # Brief stats per group
            if len(chosen_neurons) > 0:
                total_spikes = len(display_t)
                unique_display_ids, _ = np.unique(remapped_i, return_counts=True)
                active_neurons = len(unique_display_ids)
                plotted_count = len(plotted_t)
                print(f"{display_name}: {total_spikes} spikes ({active_neurons}/{len(chosen_neurons)} active), plotted {plotted_count}")

        axes[-1].set_xlabel('Time (ms)', fontsize=12)
        plt.tight_layout(pad=3.0)
        
        if save_plot:
            filename = 'improved_raster_plot.png'
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"Improved raster plot saved to '{filename}'")
        
        try:
            if os.environ.get('MPLBACKEND') == 'Agg':
                plt.close()  
            else:
                plt.show(block=True) 
        except Exception as e:
            print(f"Error displaying improved raster plot: {e}")

    except Exception as e:
        print(f"Improved raster plot Error: {str(e)}")

def plot_firing_rate_fft_multi_page(
    spike_monitors, 
    neuron_indices=None, 
    start_time=0*ms, 
    end_time=10000*ms, 
    bin_size=10*ms, 
    show_mean=True, 
    max_freq=100, 
    title='Firing Rate FFT Spectra',
    display_names=None
):
    """Plot FFT spectra of firing rates for multiple neuron groups"""
    try:
        if not spike_monitors:
            print("No spike monitors provided")
            return
            
        neuron_groups = list(spike_monitors.keys())
        n_groups = len(neuron_groups)
        
        if n_groups == 0:
            print("No neuron groups to plot")
            return
            
        cols = 4
        rows = 2
        fig, axes = plt.subplots(rows, cols, figsize=(6*cols, 4*rows))
        
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
                ax.plot(all_freqs, mean_spectrum, 'k-', linewidth=2, label='Mean')

                peak_idx = np.argmax(mean_spectrum[1:]) + 1
                peak_freq = all_freqs[peak_idx]
                peak_power = mean_spectrum[peak_idx]
                ax.axvline(peak_freq, color='red', linestyle='--', alpha=0.7, linewidth=1)
                ax.text(0.02, 0.98, f'Peak: {peak_freq:.1f} Hz', transform=ax.transAxes, 
                       verticalalignment='top', fontsize=8, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            display_name = display_names.get(group_name, group_name) if display_names else group_name
            ax.set_title(f'{display_name}', fontweight='bold', fontsize=10)
            ax.set_xlim(0, max_freq)
            ax.set_ylabel('Power', fontsize=8)
            ax.grid(True, alpha=0.3)
            
            stats_text = f'Valid: {n_valid}/{total_neurons}'
            ax.text(0.02, 0.02, stats_text, transform=ax.transAxes, 
                   verticalalignment='bottom', fontsize=7, 
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
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
        fig.suptitle(title, fontsize=16, fontweight='bold', y=0.995)
        plt.show(block=True)
        
    except Exception as e:
        print(f"Error in plot_firing_rate_fft_multi_page: {e}")
        import traceback
        traceback.print_exc()

def plot_multi_neuron_stimulus_overview(
    voltage_monitors, spike_monitors, stimulus_config,
    target_groups=['FSN', 'MSND1', 'MSND2', 'GPeT1', 'GPeTA', 'STN', 'SNr'],
    neurons_per_group=1, 
    analysis_window=(0*ms, 10000*ms),
    unified_y_scale=True, 
    threshold_clipping=True,
    display_names=None,
    thresholds=None
):
    """Plot multi-neuron membrane potentials with stimulus and spike markers"""
    try:
        available_groups = []
        
        for group_name in target_groups:
            if (group_name in voltage_monitors and 
                group_name in spike_monitors and
                len(voltage_monitors[group_name].t) > 0):
                available_groups.append(group_name)
                v_monitor = voltage_monitors[group_name]
                available_neurons_recorded = len(v_monitor.v)
                neurons_to_use = min(neurons_per_group, available_neurons_recorded)
        
        if not available_groups:
            print("No available neuron groups with voltage data or spike data for plotting.")
            return
        
        start_time, end_time = analysis_window

        max_simulation_time = 0.0 * ms
        for group_name in available_groups:
            v_monitor = voltage_monitors[group_name]
            if len(v_monitor.t) > 0:
                current_max = v_monitor.t[-1]
                max_simulation_time = max(max_simulation_time, current_max)

        if max_simulation_time < end_time:
            end_time = max_simulation_time
            print(f"Warning: Simulation time ({max_simulation_time}) is shorter than requested analysis window. Adjusting end time.")

        min_start_time = start_time
        max_end_time = end_time

        for group_name in available_groups:
            v_monitor = voltage_monitors[group_name]
            if len(v_monitor.t) > 0:
                group_start = v_monitor.t[0]
                group_end = v_monitor.t[-1]
                min_start_time = max(min_start_time, group_start)
                max_end_time = min(max_end_time, group_end)

        if max_end_time < end_time:
            end_time = max_end_time
            print(f"Warning: Available simulation time ({max_end_time}) is shorter than requested analysis window. Adjusting end time.")
        
        time_ms_float = np.arange(min_start_time/ms, max_end_time/ms + 0.1, 0.1)

        all_voltages = []
        neuron_plot_data = []
        
        for group_name in available_groups:
            v_monitor = voltage_monitors[group_name]
            s_monitor = spike_monitors[group_name]
            available_neurons_recorded = len(v_monitor.v) 
            neurons_to_use = min(neurons_per_group, available_neurons_recorded)
            
            if neurons_to_use == 1:
                selected_indices = [0]
            elif available_neurons_recorded > 0:
                selected_indices = np.linspace(0, available_neurons_recorded-1, neurons_to_use, dtype=int)
            else:
                selected_indices = []
            
            for neuron_idx in selected_indices:
                group_time_mask = (v_monitor.t >= min_start_time) & (v_monitor.t <= max_end_time)
                
                voltage_with_units = v_monitor.v[neuron_idx][group_time_mask] 

                spike_times, spike_indices = get_monitor_spikes(s_monitor)
                neuron_spike_mask = spike_indices == neuron_idx
                neuron_spike_times = spike_times[neuron_spike_mask]

                spike_time_mask = (neuron_spike_times >= min_start_time) & (neuron_spike_times <= max_end_time)
                neuron_spike_times_window = neuron_spike_times[spike_time_mask] 

                if thresholds and group_name in thresholds:
                    threshold_voltage = thresholds[group_name] * mV
                else:
                    threshold_voltage = -20 * mV

                if threshold_clipping and len(neuron_spike_times_window) > 0:
                    voltage_to_plot = voltage_with_units.copy() / mV
                    for spike_time in neuron_spike_times_window:
                        time_indices = np.where((v_monitor.t[group_time_mask] >= spike_time - 0.1*ms) & 
                                               (v_monitor.t[group_time_mask] <= spike_time + 0.1*ms))[0]
                        if len(time_indices) > 0:
                            voltage_to_plot[time_indices] = threshold_voltage / mV
                else:
                    voltage_to_plot = voltage_with_units / mV

                neuron_plot_data.append({
                    'group_name': group_name,
                    'neuron_idx': neuron_idx,
                    'voltage': voltage_to_plot,
                    'spike_times': neuron_spike_times_window / ms,
                    'threshold': threshold_voltage / mV
                })

                all_voltages.extend(voltage_to_plot)
        
        if unified_y_scale and all_voltages:
            y_min = np.min(all_voltages) - 5.0
            y_max = np.max(all_voltages) + 15.0  # Increased to accommodate spike markers
            y_range = (y_min, y_max)
        else:
            y_range = None
        
        fig_height = 3 + len(neuron_plot_data) * 1.5
        fig, axes = plt.subplots(len(neuron_plot_data) + 1, 1, figsize=(16, fig_height), 
                                sharex=True, gridspec_kw={'height_ratios': [1] + [1]*len(neuron_plot_data)})
        
        stimulus_pA = np.zeros_like(time_ms_float)
        
        if stimulus_config and stimulus_config.get('enabled', False):
            stim_start = stimulus_config.get('start_time', 0*ms)
            stim_duration = stimulus_config.get('duration', 0*ms)
            
            if not isinstance(stim_start, Quantity):
                stim_start = stim_start * ms
            if not isinstance(stim_duration, Quantity):
                stim_duration = stim_duration * ms

            stim_end = stim_start + stim_duration
            
            stim_mask = (time_ms_float * ms >= stim_start) & (time_ms_float * ms <= stim_end)
            stimulus_amplitude = stimulus_config.get('amplitude', 25)
            stimulus_pA[stim_mask] = stimulus_amplitude
        
        axes[0].plot(time_ms_float, stimulus_pA, 'k-', linewidth=2)
        axes[0].set_ylabel('Stimulus (pA)', fontsize=12, fontweight='bold')
        axes[0].set_title('Multi-Neuron Membrane Potential', 
                         fontsize=16, fontweight='bold', pad=20)
        axes[0].set_ylim(-5, max(stimulus_pA) + 10 if max(stimulus_pA) > 0 else 30)
        
        if stimulus_config and stimulus_config.get('enabled', False):
            axes[0].axvspan(stim_start/ms, stim_end/ms, alpha=0.15, color='orange', label='Stimulus Period')
            axes[0].legend(loc='upper right', fontsize=9)
        
        for ax_idx, data in enumerate(neuron_plot_data):
            ax = axes[ax_idx + 1]
            
            group_name = data['group_name']
            neuron_idx = data['neuron_idx']
            voltage = data['voltage']
            spike_times = data['spike_times']
            threshold = data['threshold']
            
            actual_time_for_plot = v_monitor.t[group_time_mask] / ms 
            
            ax.plot(actual_time_for_plot, voltage, 'b-', linewidth=1.2, alpha=0.8)
            
            if len(spike_times) > 0:
                spike_voltages = [threshold + 2] * len(spike_times)
                
                ax.scatter(spike_times, spike_voltages, color='red', s=15, 
                          marker='o', alpha=0.9, zorder=5, edgecolors='black', linewidth=0.5)
                
                if threshold_clipping:
                    ax.axhline(threshold, color='orange', linestyle='--', alpha=0.7, linewidth=2)
            else:
                if threshold_clipping:
                    ax.axhline(threshold, color='orange', linestyle='--', alpha=0.7, linewidth=2)
            
            if unified_y_scale and y_range:
                ax.set_ylim(y_range)
            
            display_name = display_names.get(group_name, group_name) if display_names else group_name
            ax.set_ylabel(f'{display_name}\n#{neuron_idx}\n(mV)', fontsize=10, fontweight='bold')
            ax.grid(True, alpha=0.2)
            
            ax.set_xlim(min_start_time/ms, max_end_time/ms)
            
            spike_count = len(spike_times)
            if spike_count > 0:
                ax.text(0.98, 0.95, f'{spike_count} spikes', transform=ax.transAxes, 
                       ha='right', va='top', fontsize=9, 
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
            else:
                ax.text(0.98, 0.95, 'No spikes', transform=ax.transAxes, 
                       ha='right', va='top', fontsize=9, color='red',
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
        
        axes[-1].set_xlabel('Time (ms)', fontsize=12, fontweight='bold')
        axes[-1].set_xlim(min_start_time/ms, max_end_time/ms) 

        plt.tight_layout()
        
        print(f"Analysis period: {start_time/ms:.0f} - {end_time/ms:.0f} ms")
        print(f"Total neurons displayed: {len(neuron_plot_data)}")
        
        if unified_y_scale and y_range:
            print(f"Y-axis range: {y_range[0]:.1f} to {y_range[1]:.1f} mV")
        
        for data in neuron_plot_data:
            group_name = data['group_name']
            neuron_idx = data['neuron_idx']
            spike_count = len(data['spike_times'])
            threshold = data['threshold']
            display_name = display_names.get(group_name, group_name) if display_names else group_name
            print(f"{display_name} #{neuron_idx}: {spike_count} spikes, threshold: {threshold:.1f} mV")
        
        try:
            plt.show(block=True) 
        except Exception as e:
            print(f"Error displaying enhanced multi-neuron overview: {e}")
            
    except Exception as e:
        print(f"Enhanced multi-neuron overview error: {str(e)}")
        import traceback
        traceback.print_exc()

def plot_continuous_firing_rate_with_samples(spike_monitors, start_time=0*ms, end_time=10000*ms, bin_size=20*ms, 
                                            plot_order=None, display_names=None, stimulus_config=None, 
                                            smooth_sigma=3, save_plot=True, n_samples=10, neurons_per_sample=30):
    """Plot continuous firing rates with multiple samples and smoothing"""
    def calculate_firing_rate_for_neuron_subset(spike_times, spike_indices, 
                                              selected_neurons, total_neurons, time_bins):
        firing_rates = []
        for i in range(len(time_bins) - 1):
            bin_start = time_bins[i]
            bin_end = time_bins[i + 1]
            time_mask = (spike_times >= bin_start) & (spike_times < bin_end)
            neuron_mask = np.isin(spike_indices, selected_neurons)
            combined_mask = time_mask & neuron_mask
            spike_count = np.sum(combined_mask)
            bin_duration = (bin_end - bin_start) / 1000.0
            rate = spike_count / (len(selected_neurons) * bin_duration)
            firing_rates.append(rate)
        return np.array(firing_rates)
    try:
        if plot_order:
            spike_monitors = {name: spike_monitors[name] for name in plot_order if name in spike_monitors}
        else:
            plot_order = list(spike_monitors.keys())
        if not spike_monitors:
            print("No valid neuron groups to plot.")
            return
        time_bins = np.arange(start_time/ms, end_time/ms + bin_size/ms, bin_size/ms)
        time_centers = time_bins[:-1] + bin_size/ms/2
        neuron_names = list(spike_monitors.keys())
        n_groups = len(neuron_names)
        cols = 4
        rows = 2
        fig, axes = plt.subplots(rows, cols, figsize=(6*cols, 4*rows))

        if n_groups == 1:
            axes = [axes]
        elif rows == 1:
            axes = axes.reshape(1, -1)
        
        colors = ['#E74C3C', '#8E44AD', '#3498DB', '#27AE60', '#F39C12', '#34495E', '#E67E22']
        np.random.seed(2025)
        for idx, name in enumerate(neuron_names):
            row = idx // cols
            col = idx % cols
            ax = axes[row, col] if rows > 1 else axes[col]
            print(f"Debug: Processing {name} (idx={idx})")
            if name not in spike_monitors:
                print(f"Warning: {name} not found in spike_monitors")
                continue
            try:
                spike_times, spike_indices = get_monitor_spikes(spike_monitors[name])
                print(f"Debug: {name} - got {len(spike_times)} spikes")
                if len(spike_times) == 0:
                    ax.text(0.5, 0.5, f'{name}\nNo spikes', 
                           transform=ax.transAxes, ha='center', va='center', fontsize=12)
                    ax.set_title(f'{name}', fontweight='bold')
                    continue
            except Exception as e:
                print(f"Error processing {name}: {e}")
                ax.text(0.5, 0.5, f'{name}\nError: {str(e)[:50]}', 
                       transform=ax.transAxes, ha='center', va='center', fontsize=12)
                ax.set_title(f'{name}', fontweight='bold')
                continue
            spike_times_ms = spike_times / ms
            total_neurons = spike_monitors[name].source.N
            time_mask = (spike_times_ms >= start_time/ms) & (spike_times_ms <= end_time/ms)
            spike_times_window = spike_times_ms[time_mask]
            spike_indices_window = spike_indices[time_mask]
            max_start_neuron = max(0, total_neurons - neurons_per_sample)
            sample_firing_rates = []
            for sample_idx in range(n_samples):
                if max_start_neuron <= 0:
                    selected_neurons = np.arange(min(neurons_per_sample, total_neurons))
                else:
                    start_neuron = np.random.randint(0, max_start_neuron + 1)
                    selected_neurons = np.arange(start_neuron, 
                                                min(start_neuron + neurons_per_sample, total_neurons))
                firing_rate = calculate_firing_rate_for_neuron_subset(
                    spike_times_window, spike_indices_window, selected_neurons, 
                    total_neurons, time_bins)
                sample_firing_rates.append(firing_rate)
            sample_firing_rates = np.array(sample_firing_rates)
            avg_firing_rate = np.mean(sample_firing_rates, axis=0)
            std_firing_rate = np.std(sample_firing_rates, axis=0)
            
            firing_rate_smooth = gaussian_smooth(avg_firing_rate, smooth_sigma)
            std_firing_rate_smooth = gaussian_smooth(std_firing_rate, smooth_sigma)
            
            ax.fill_between(time_centers, 
                          firing_rate_smooth - std_firing_rate_smooth, 
                          firing_rate_smooth + std_firing_rate_smooth, 
                          alpha=0.3, color=colors[idx % len(colors)], 
                          label=f'{name} ±1σ')
            ax.plot(time_centers, firing_rate_smooth, color=colors[idx % len(colors)], 
                   linewidth=2, label=f'{name} (avg)')
            
            max_rate = np.max(firing_rate_smooth)
            min_rate = np.min(firing_rate_smooth)
            avg_std = np.mean(std_firing_rate_smooth)
            stats_text = f'Max: {max_rate:.1f} Hz\nMin: {min_rate:.1f} Hz\nAvg Std: {avg_std:.1f} Hz'
            ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
                   verticalalignment='top', fontsize=8, 
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            ax.set_title(f'{name}', fontweight='bold')
            ax.set_ylabel('Firing Rate (Hz)', fontsize=11)
            ax.set_xlim([start_time/ms, end_time/ms])
            ax.legend(fontsize=9)
            ax.grid(True, linestyle='--', alpha=0.5)
        
        for idx in range(n_groups, rows * cols):
            if rows > 1:
                row = idx // cols
                col = idx % cols
                axes[row, col].set_visible(False)
            elif cols > 1:
                axes[idx].set_visible(False)
        for ax in axes.flat if hasattr(axes, 'flat') else axes:
            ax.set_xlabel('Time (ms)', fontsize=12, fontweight='bold')
        plt.tight_layout(rect=[0, 0, 1, 0.97])
        fig.suptitle('Average Firing Rate with Variance', fontsize=16, fontweight='bold', y=0.995)
        if save_plot:
            filename = 'continuous_firing_rate_multi_sample_avg_grid.png'
            plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white')
            print(f"Continuous firing rate saved to '{filename}'")
        try:
            plt.show(block=True)
        except Exception as e:
            print(f"Error displaying continuous firing rate: {e}")
    except Exception as e:
        print(f"Continuous firing rate error: {str(e)}")
        import traceback
        traceback.print_exc()

def plot_membrane_zoom(voltage_monitors, time_window=(0*ms, 100*ms), plot_order=None, neuron_indices=None, group_name=None, spike_monitors=None, thresholds=None, display_names=None):
    """Plot zoomed membrane potential with spike clipping"""
    try:
        if hasattr(voltage_monitors, 'items'):
            if plot_order:
                filtered_monitors = {name: voltage_monitors[name] for name in plot_order if name in voltage_monitors}
            else:
                filtered_monitors = voltage_monitors

            # Select only the first neuron group
            first_group = list(filtered_monitors.keys())[0]
            monitor = filtered_monitors[first_group]
            
            t = monitor.t / ms
            if neuron_indices is not None and len(neuron_indices) > 0:
                v = monitor.v[neuron_indices[0]] / mV 
                neuron_idx = neuron_indices[0]
            else:
                v = monitor.v[0] / mV 
                neuron_idx = 0
            
            mask = (t >= time_window[0]/ms) & (t <= time_window[1]/ms)
            
            spike_times = []
            if spike_monitors and first_group in spike_monitors:
                spike_monitor = spike_monitors[first_group]
                spike_t, spike_i = get_monitor_spikes(spike_monitor)
                neuron_spike_mask = spike_i == neuron_idx
                neuron_spike_times = spike_t[neuron_spike_mask]
                spike_time_mask = (neuron_spike_times >= time_window[0]) & (neuron_spike_times <= time_window[1])
                spike_times = neuron_spike_times[spike_time_mask] / ms
            
            threshold = -20
            if thresholds and first_group in thresholds:
                threshold = thresholds[first_group]
            
            voltage_to_plot = v[mask].copy()
            
            if len(spike_times) > 0:
                for spike_time in spike_times:
                    time_indices = np.where((t[mask] >= spike_time - 0.1) & (t[mask] <= spike_time + 0.1))[0]
                    if len(time_indices) > 0:
                        voltage_to_plot[time_indices] = threshold
            
            plt.figure(figsize=(12, 4)) 
            plt.plot(t[mask], voltage_to_plot, linewidth=1.5, color='#2E86AB')
            
            display_name = display_names.get(first_group, first_group) if display_names else first_group
            plt.title(f"{display_name} Membrane Potential Zoom ({time_window[0]/ms:.0f}-{time_window[1]/ms:.0f} ms)", 
                     fontsize=14, fontweight='bold')
            plt.xlabel("Time (ms)", fontsize=12)
            plt.ylabel("V (mV)", fontsize=12)
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.show(block=True)
        else:
            monitor = voltage_monitors
            t = monitor.t / ms
            if neuron_indices is not None and len(neuron_indices) > 0:
                v = monitor.v[neuron_indices[0]] / mV
                neuron_idx = neuron_indices[0]
            else:
                v = monitor.v[0] / mV
                neuron_idx = 0
            mask = (t >= time_window[0]/ms) & (t <= time_window[1]/ms)
            name = group_name if group_name is not None else "Neuron"

            spike_times = []
            if spike_monitors and group_name in spike_monitors:
                spike_monitor = spike_monitors[group_name]
                spike_t, spike_i = get_monitor_spikes(spike_monitor)
                neuron_spike_mask = spike_i == neuron_idx
                neuron_spike_times = spike_t[neuron_spike_mask]
                spike_time_mask = (neuron_spike_times >= time_window[0]) & (neuron_spike_times <= time_window[1])
                spike_times = neuron_spike_times[spike_time_mask] / ms
            
            threshold = -20
            if thresholds and group_name in thresholds:
                threshold = thresholds[group_name]

            voltage_to_plot = v[mask].copy()
            
            if len(spike_times) > 0:
                for spike_time in spike_times:
                    time_indices = np.where((t[mask] >= spike_time - 0.1) & (t[mask] <= spike_time + 0.1))[0]
                    if len(time_indices) > 0:
                        voltage_to_plot[time_indices] = threshold
                        spike_idx = np.argmin(np.abs(t[mask] - spike_time))
                        if spike_idx < len(voltage_to_plot):
                            voltage_to_plot[spike_idx] = threshold + 2
                        spike_idx = np.argmin(np.abs(t[mask] - spike_time))
                        if spike_idx < len(voltage_to_plot):
                            voltage_to_plot[spike_idx] = threshold + 2
            
            plt.figure(figsize=(12, 4)) 
            plt.plot(t[mask], voltage_to_plot, linewidth=1.5, color='#2E86AB')
            
            display_name = display_names.get(name, name) if display_names else name
            plt.title(f"{display_name} Membrane Potential Zoom ({time_window[0]/ms:.0f}-{time_window[1]/ms:.0f} ms)", 
                    fontsize=14, fontweight='bold')
            plt.xlabel("Time (ms)", fontsize=12)
            plt.ylabel("V (mV)", fontsize=12)
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.show(block=True)
    except Exception as e:
        print(f"Error in plot_membrane_zoom: {e}")
        import traceback
        traceback.print_exc()

def plot_raster_zoom(spike_monitor, time_window=(0*ms, 100*ms), neuron_indices=None, group_name=None, display_names=None):
    """Plot zoomed raster plot for spike visualization"""
    t = spike_monitor.t / ms
    i = spike_monitor.i
    mask = (t >= time_window[0]/ms) & (t <= time_window[1]/ms)
    t_zoom = t[mask]
    i_zoom = i[mask]
    if neuron_indices is not None:
        neuron_mask = np.isin(i_zoom, neuron_indices)
        t_zoom = t_zoom[neuron_mask]
        i_zoom = i_zoom[neuron_mask]
    name = group_name if group_name is not None else "Neuron"
    display_name = display_names.get(name, name) if display_names else name
    plt.figure(figsize=(12, 6))
    plt.scatter(t_zoom, i_zoom, s=3, alpha=0.7, color='#E74C3C')
    plt.title(f"{display_name} Raster Zoom ({time_window[0]/ms:.0f}-{time_window[1]/ms:.0f} ms)", 
             fontsize=14, fontweight='bold')
    plt.xlabel("Time (ms)", fontsize=12)
    plt.ylabel("Neuron Index", fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show(block=True)

def analyze_input_rates_and_spike_counts(spike_monitors, external_inputs, neuron_configs, stimulus_config=None, analysis_start_time=2000*ms, analysis_end_time=10000*ms):
    """Analyze input rates and compare with actual spike counts"""
    print("\n" + "="*60)
    print("FIRING RATE ANALYSIS")
    print("="*60)
    
    input_rates = {}
    for neuron_config in neuron_configs:
        if neuron_config.get('neuron_type') == 'poisson':
            name = neuron_config['name']
            if 'target_rates' in neuron_config:
                target, rate_info = list(neuron_config['target_rates'].items())[0]
                rate_expr = rate_info['equation']
                
                if '*Hz' in rate_expr:
                    base_rate = eval(rate_expr)
                else:
                    base_rate = eval(rate_expr) * Hz
                
                N = neuron_config['N']
                rate_per_neuron = base_rate / N
                
                input_rates[target] = {
                    'total_rate': base_rate,
                    'rate_per_neuron': rate_per_neuron,
                    'N': N,
                    'source': name
                }
                
                print(f"[{target}] Input: {rate_per_neuron/Hz:.3f} Hz/neuron")
                
                if stimulus_config and stimulus_config.get('enabled', False):
                    if target in stimulus_config.get('rates', {}):
                        stim_rate_total = stimulus_config['rates'][target] * Hz
                        stim_rate_per_neuron = stim_rate_total / N
                        
                        stim_start = stimulus_config.get('start_time', 0)
                        stim_duration = stimulus_config.get('duration', 0)
                        stim_end = stim_start + stim_duration
                        
                        print(f"  Stimulus: {stim_rate_per_neuron/Hz:.3f} Hz/neuron (+{((stim_rate_per_neuron - rate_per_neuron) / rate_per_neuron * 100):+.1f}%)")
    
    print(f"\nAnalysis period: {analysis_start_time/ms:.0f}-{analysis_end_time/ms:.0f}ms")
    for name, monitor in spike_monitors.items():
        spike_times, spike_indices = get_monitor_spikes(monitor)
        
        if len(spike_times) == 0:
            print(f"\n[{name}]: No spikes recorded")
            continue
        
        total_neurons = monitor.source.N
        
        time_mask = (spike_times >= analysis_start_time) & (spike_times <= analysis_end_time)
        analysis_spikes = spike_times[time_mask]
        analysis_indices = spike_indices[time_mask]
        
        total_spikes = len(analysis_spikes)
        analysis_duration = (analysis_end_time - analysis_start_time) / second
        
        overall_rate = total_spikes / (total_neurons * analysis_duration)
        
        individual_rates = []
        active_neurons = 0
        
        for neuron_idx in range(total_neurons):
            neuron_spikes = np.sum(analysis_indices == neuron_idx)
            neuron_rate = neuron_spikes / analysis_duration
            individual_rates.append(neuron_rate)
            if neuron_spikes > 0:
                active_neurons += 1
        
        individual_rates = np.array(individual_rates)
        mean_rate = np.mean(individual_rates)
        std_rate = np.std(individual_rates)
        max_rate = np.max(individual_rates)
        min_rate = np.min(individual_rates)
        
        print(f"[{name}] Rate: {mean_rate:.3f} Hz (active: {active_neurons}/{total_neurons}, max: {max_rate:.1f} Hz)")
        
        if name in input_rates:
            expected_rate = input_rates[name]['rate_per_neuron'] / Hz
            rate_ratio = mean_rate / expected_rate if expected_rate > 0 else 0
            
            if rate_ratio > 2.0:
                print(f"  ⚠️  Rate {rate_ratio:.1f}x higher than expected")
            elif rate_ratio < 0.5:
                print(f"  ⚠️  Rate {1/rate_ratio:.1f}x lower than expected")
            
            if active_neurons/total_neurons < 0.1:
                print(f"  ⚠️  Only {active_neurons/total_neurons*100:.1f}% neurons active")
    
    print("\n" + "="*60)

def plot_multi_neuron_membrane_potential_comparison(voltage_monitors, spike_monitors, 
                                                   target_groups=['FSN', 'MSND1', 'MSND2', 'GPeT1', 'GPeTA', 'STN', 'SNr'],
                                                   neurons_per_group=5, 
                                                   analysis_window=(2000*ms, 10000*ms),
                                                   display_names=None,
                                                   thresholds=None):
    """Plot multiple neurons' membrane potential with spike markers"""
    try:
        available_groups = []
        
        for group_name in target_groups:
            if (group_name in voltage_monitors and 
                group_name in spike_monitors and
                len(voltage_monitors[group_name].t) > 0):
                available_groups.append(group_name)
        
        if not available_groups:
            print("No available neuron groups with voltage data for plotting.")
            return
        
        start_time, end_time = analysis_window
        
        # Plot multiple neurons' membrane potential for each group
        fig, axes = plt.subplots(len(available_groups), 1, figsize=(16, 4 * len(available_groups)), sharex=True)
        if len(available_groups) == 1:
            axes = [axes]
        
        for idx, group_name in enumerate(available_groups):
            ax = axes[idx]
            v_monitor = voltage_monitors[group_name]
            s_monitor = spike_monitors[group_name]
            
            recorded_neurons = v_monitor.record
            total_recorded = len(recorded_neurons)
            neurons_to_plot = min(neurons_per_group, total_recorded)
            
            if neurons_to_plot == 1:
                selected_indices = [0]
            else:
                selected_indices = np.linspace(0, total_recorded-1, neurons_to_plot, dtype=int)
            
            time_mask = (v_monitor.t >= start_time) & (v_monitor.t <= end_time)
            plot_times = v_monitor.t[time_mask] / ms
            
            colors = plt.cm.viridis(np.linspace(0, 1, len(selected_indices)))
            
            for i, neuron_idx in enumerate(selected_indices):
                voltage = v_monitor.v[neuron_idx][time_mask] / mV
                
                spike_times, spike_indices = get_monitor_spikes(s_monitor)
                neuron_spike_mask = spike_indices == neuron_idx
                neuron_spike_times = spike_times[neuron_spike_mask]
                spike_time_mask = (neuron_spike_times >= start_time) & (neuron_spike_times <= end_time)
                neuron_spike_times_window = neuron_spike_times[spike_time_mask] / ms
                
                voltage_clipped = voltage.copy()
                threshold = thresholds.get(group_name, -20) if thresholds else -20
                
                if len(neuron_spike_times_window) > 0:
                    for spike_time in neuron_spike_times_window:
                        time_indices = np.where((plot_times >= spike_time - 0.1) & (plot_times <= spike_time + 0.1))[0]
                        if len(time_indices) > 0:
                            voltage_clipped[time_indices] = threshold
                
                neuron_spike_count = len(neuron_spike_times_window)
                
                if len(neuron_spike_times_window) > 0 or i < 3:
                    ax.plot(plot_times, voltage_clipped, color=colors[i], alpha=0.7, linewidth=1, 
                           label=f'Neuron {neuron_idx} ({neuron_spike_count} spikes)')
                else:
                    ax.plot([], [], color=colors[i], alpha=0.7, linewidth=1, 
                           label=f'Neuron {neuron_idx} ({neuron_spike_count} spikes)')
                
                if len(neuron_spike_times_window) > 0:
                    spike_voltages = [threshold + 2] * len(neuron_spike_times_window)
                    ax.scatter(neuron_spike_times_window, spike_voltages, 
                             color=colors[i], s=40, marker='^', alpha=0.9, zorder=5, edgecolors='black', linewidth=1)
                else:
                    if i < 3:
                        ax.text(0.02, 0.98 - i*0.1, f'Neuron {neuron_idx}: No spikes', 
                               transform=ax.transAxes, fontsize=8, color=colors[i], 
                               verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
            
            if thresholds and group_name in thresholds:
                threshold = thresholds[group_name]
                ax.axhline(threshold, color='red', linestyle='--', alpha=0.7, linewidth=2, label=f'Threshold ({threshold} mV)')
            else:
                ax.axhline(-20, color='red', linestyle='--', alpha=0.7, linewidth=2, label='Threshold (-20 mV)')
            
            display_name = display_names.get(group_name, group_name) if display_names else group_name
            group_spike_times, group_spike_indices = get_monitor_spikes(s_monitor)
            group_time_mask = (group_spike_times >= start_time) & (group_spike_times <= end_time)
            total_group_spikes = len(group_spike_times[group_time_mask])
            ax.set_title(f'{display_name} - Multiple Neurons Membrane Potential (Total Spikes: {total_group_spikes})', fontsize=14, fontweight='bold')
            ax.set_ylabel('Membrane Potential (mV)', fontsize=12)
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=10, loc='upper right')
            
            ax.set_ylim(-80, 60)
        
        axes[-1].set_xlabel('Time (ms)', fontsize=12)
        axes[-1].set_xlim(start_time/ms, end_time/ms)
        
        plt.tight_layout()
        plt.show(block=True)
        
        print(f"\nMembrane Potential Analysis: {start_time/ms:.0f}-{end_time/ms:.0f}ms")
        for group_name in available_groups:
            s_monitor = spike_monitors[group_name]
            spike_times, spike_indices = get_monitor_spikes(s_monitor)
            
            time_mask = (spike_times >= start_time) & (spike_times <= end_time)
            analysis_spikes = spike_times[time_mask]
            analysis_indices = spike_indices[time_mask]
            
            total_neurons = s_monitor.source.N
            total_spikes = len(analysis_spikes)
            analysis_duration = (end_time - start_time) / second
            overall_rate = total_spikes / (total_neurons * analysis_duration)
            
            display_name = display_names.get(group_name, group_name) if display_names else group_name
            print(f"{display_name}: {overall_rate:.3f} Hz ({total_spikes} spikes)")
        
    except Exception as e:
        print(f"Error in plot_multi_neuron_membrane_potential_comparison: {e}")
        import traceback
        traceback.print_exc()

def compute_pca_trajectories(spike_monitors, bin_size=10*ms, cycle_ms=1000, 
                             stimulus_start_ms=0, groups=None, n_pcs=3,
                             start_time=0*ms, end_time=10000*ms,
                             display_names=None):
    """Compute PCA trajectories using binned population activity.

    - spike_monitors: dict[name -> SpikeMonitor]
    - bin_size: width of time bin for population rates
    - cycle_ms: length of one cycle after stimulus (e.g., 1000ms)
    - stimulus_start_ms: cycle origin (ms)
    - groups: subset of groups to include (default: all in spike_monitors)
    - n_pcs: number of principal components to return
    - returns: dict with keys: time_centers_ms, pcs (T x n_pcs), per_cycle_indices
    """
    try:
        if groups is None:
            groups = list(spike_monitors.keys())
        groups = [g for g in groups if g in spike_monitors]
        if len(groups) == 0:
            return None

        t0 = max(start_time, stimulus_start_ms*ms)
        t1 = end_time
        bin_edges = np.arange(t0/ms, t1/ms + bin_size/ms, bin_size/ms)
        time_centers = bin_edges[:-1] + (bin_size/ms)/2

        # Build population vector: concatenated per-group binned rates
        pop_vectors = []  # list of arrays length = num_bins
        for g in groups:
            sm = spike_monitors[g]
            st = sm.t / ms
            si = sm.i
            N = sm.source.N
            mask = (st >= t0/ms) & (st <= t1/ms)
            st_w = st[mask]
            si_w = si[mask]

            # per-neuron histogram → sum → rate per group
            rates_group = np.zeros(len(bin_edges)-1)
            if len(st_w) > 0:
                # accumulate per-neuron counts per bin
                for idx_bin in range(len(bin_edges)-1):
                    b0 = bin_edges[idx_bin]
                    b1 = bin_edges[idx_bin+1]
                    m = (st_w >= b0) & (st_w < b1)
                    counts = np.sum(m)
                    duration_s = (b1 - b0) / 1000.0
                    rates_group[idx_bin] = counts / (N * duration_s)
            pop_vectors.append(rates_group)

        # concatenate groups into population activity matrix (time x features)
        X = np.stack(pop_vectors, axis=1)  # shape (T, G)
        X = X - np.mean(X, axis=0, keepdims=True)

        # PCA via SVD
        U, S, Vt = np.linalg.svd(X, full_matrices=False)
        PCs = U[:, :n_pcs] * S[:n_pcs]

        # cycle indices for plotting later
        rel_ms = time_centers - stimulus_start_ms
        rel_ms[rel_ms < 0] = np.nan
        per_cycle = np.floor(rel_ms / cycle_ms).astype(float)

        return {
            'time_centers_ms': time_centers,
            'pcs': PCs,
            'groups': groups,
            'per_cycle_indices': per_cycle
        }
    except Exception as e:
        print(f"Error in compute_pca_trajectories: {e}")
        return None

def plot_pca_trajectories(pca_result, cycles_to_show=6, colors=None, title='PCA Trajectories'):
    """Plot PCA trajectories in 3D and per-cycle 2D projections.
    pca_result: output of compute_pca_trajectories
    """
    try:
        if pca_result is None:
            print('No PCA result to plot')
            return
        time_centers = pca_result['time_centers_ms']
        PCs = pca_result['pcs']  # (T, k)
        per_cycle = pca_result['per_cycle_indices']
        k = PCs.shape[1]
        if k < 3:
            print('Less than 3 PCs, plotting first 2 only')
        if colors is None:
            colors = ['#F5B041', '#DC7633', '#6E2C00', '#A04000', '#CA6F1E', '#7E5109']

        # 3D trajectory (first 3 PCs)
        fig = plt.figure(figsize=(12, 5))
        ax3d = fig.add_subplot(1, 2, 1, projection='3d')
        valid_mask = ~np.isnan(per_cycle)
        ax3d.plot(PCs[valid_mask,0], PCs[valid_mask,1], PCs[valid_mask,2 if k>=3 else 1], color='black', linewidth=2)
        ax3d.set_xlabel('PC 1')
        ax3d.set_ylabel('PC 2')
        ax3d.set_zlabel('PC 3')
        ax3d.set_title(title)

        # per-cycle loops in 2D (PC1-2)
        ax_grid = fig.add_subplot(1, 2, 2)
        for c in range(int(np.nanmin(per_cycle)), int(np.nanmin([np.nanmax(per_cycle), cycles_to_show-1]))+1):
            m = (per_cycle == c)
            if np.any(m):
                col = colors[c % len(colors)]
                ax_grid.plot(PCs[m,0], PCs[m,1], color=col, linewidth=2)
        ax_grid.set_xlabel('PC 1')
        ax_grid.set_ylabel('PC 2')
        ax_grid.set_title('Cycle-specific trajectories (PC1–PC2)')
        ax_grid.axis('equal')
        plt.tight_layout()
        plt.show(block=True)
    except Exception as e:
        print(f"Error in plot_pca_trajectories: {e}")