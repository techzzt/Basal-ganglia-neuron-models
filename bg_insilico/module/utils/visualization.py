import matplotlib
import matplotlib.pyplot as plt
import os
import numpy as np 
from brian2 import *
import platform

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
                                stimulus_periods=None, save_plot=True):

    np.random.seed(2025)
    try:
        if plot_order:
            spike_monitors = {name: spike_monitors[name] for name in plot_order if name in spike_monitors}

        if not spike_monitors:
            print("No valid neuron groups to plot.")
            return
        
        n_plots = len(spike_monitors)
        fig, axes = plt.subplots(n_plots, 1, figsize=(18, 3.5 * n_plots), sharex=True)
        if n_plots == 1:
            axes = [axes]

        print(f"\nImproved Raster plot Range: {start_time/ms:.0f}ms - {end_time/ms:.0f}ms")
          
        for i, (name, monitor) in enumerate(spike_monitors.items()):
            spike_times, spike_indices = get_monitor_spikes(monitor)
            
            if len(spike_indices) == 0:
                print(f"No spikes recorded for {name}")
                continue
            
            total_neurons = monitor.source.N
            chosen_neurons = np.random.choice(total_neurons, size=min(sample_size, total_neurons), replace=False)
            chosen_neurons = sorted(chosen_neurons)

            time_mask = (spike_times >= start_time) & (spike_times <= end_time)
            neuron_mask = np.isin(spike_indices, chosen_neurons)
            combined_mask = time_mask & neuron_mask

            display_t = spike_times[combined_mask]
            display_i = spike_indices[combined_mask]

            neuron_mapping = {original: new for new, original in enumerate(chosen_neurons)}
            remapped_i = [neuron_mapping[original] for original in display_i]

            display_name = display_names.get(name, name) if display_names else name
            
            axes[i].scatter(display_t / ms, remapped_i, s=5.0, alpha=0.9, edgecolors='black', linewidth=0.3)
            axes[i].set_title(f'{display_name} Raster Plot', fontsize=14, pad=15)
            axes[i].set_ylabel('')
            axes[i].set_yticks([])
            
            if len(chosen_neurons) > 0:
                axes[i].set_ylim(-0.5, len(chosen_neurons) - 0.5)
                if len(chosen_neurons) <= 15: 
                    axes[i].set_yticks(range(len(chosen_neurons)))
                else: 
                    tick_indices = range(0, len(chosen_neurons), max(1, len(chosen_neurons)//5))
                    axes[i].set_yticks(tick_indices)
            axes[i].set_xlim(int(start_time/ms), int(end_time/ms))
            
            for j in range(0, len(chosen_neurons), max(1, len(chosen_neurons)//10)):
                axes[i].axhline(y=j-0.5, color='gray', alpha=0.2, linewidth=0.3)
            
            if stimulus_periods:
                for period_idx, (stim_start, stim_end) in enumerate(stimulus_periods):
                    if stim_start >= start_time and stim_end <= end_time:
                        axes[i].axvspan(stim_start/ms, stim_end/ms, alpha=0.2, color='red')
                
                axes[i].grid(True, alpha=0.08, axis='x')
            
            print(f"{display_name}: {len(display_t)} spikes shown (sampled from {sample_size} neurons)")

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
                plt.show(block=False) 
                plt.pause(0.1) 
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
            neurons_with_spikes = 0
            
            for neuron_idx in neuron_indices:
                neuron_spikes = spike_times[spike_indices == neuron_idx]
                if len(neuron_spikes) == 0:
                    continue
                
                neurons_with_spikes += 1
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
        plt.show()
        
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
    try:
        available_groups = []
        total_neurons = 0
        
        for group_name in target_groups:
            if (group_name in voltage_monitors and 
                group_name in spike_monitors and
                len(voltage_monitors[group_name].t) > 0):
                available_groups.append(group_name)
                v_monitor = voltage_monitors[group_name]
                available_neurons = v_monitor.N
                available_neurons_recorded = len(v_monitor.v)
                neurons_to_use = min(neurons_per_group, available_neurons_recorded)
                total_neurons += neurons_to_use
        
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
                    voltage_to_plot = np.minimum(voltage_with_units, threshold_voltage) / mV
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
            y_max = np.max(all_voltages) + 10.0
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
                spike_voltages = [threshold] * len(spike_times)
                
                ax.scatter(spike_times, spike_voltages, color='red', s=12, 
                          marker='o', alpha=0.8, zorder=5)
                
                if threshold_clipping:
                    ax.axhline(threshold, color='orange', linestyle='--', alpha=0.5, linewidth=1)
            
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
        
        axes[-1].set_xlabel('Time (ms)', fontsize=12, fontweight='bold')
        axes[-1].set_xlim(min_start_time/ms, max_end_time/ms) 

        plt.tight_layout()
        plt.show()
        
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
            if name not in spike_monitors:
                print(f"Warning: {name} not found in spike_monitors")
                continue
            spike_times, spike_indices = get_monitor_spikes(spike_monitors[name])
            if len(spike_times) == 0:
                ax.text(0.5, 0.5, f'{name}\nNo spikes', 
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
    try:
        if hasattr(voltage_monitors, 'items'):
            if plot_order:
                filtered_monitors = {name: voltage_monitors[name] for name in plot_order if name in voltage_monitors}
            else:
                filtered_monitors = voltage_monitors

            all_voltages = []
            for name, monitor in filtered_monitors.items():
                t = monitor.t / ms
                if neuron_indices is not None and len(neuron_indices) > 0:
                    v = monitor.v[neuron_indices[0]] / mV 
                else:
                    v = monitor.v[0] / mV 
                mask = (t >= time_window[0]/ms) & (t <= time_window[1]/ms)
                if np.any(mask):
                    all_voltages.extend(v[mask])
            
            if all_voltages:
                y_min = np.min(all_voltages) - 5.0
                y_max = np.max(all_voltages) + 10.0
                y_range = (y_min, y_max)
            else:
                y_range = None

            for name, monitor in filtered_monitors.items():
                t = monitor.t / ms
                if neuron_indices is not None and len(neuron_indices) > 0:
                    v = monitor.v[neuron_indices[0]] / mV 
                    neuron_idx = neuron_indices[0]
                else:
                    v = monitor.v[0] / mV 
                    neuron_idx = 0
                mask = (t >= time_window[0]/ms) & (t <= time_window[1]/ms)
                
                # Get spike times for this neuron if spike_monitors provided
                spike_times = []
                if spike_monitors and name in spike_monitors:
                    spike_monitor = spike_monitors[name]
                    spike_t, spike_i = get_monitor_spikes(spike_monitor)
                    neuron_spike_mask = spike_i == neuron_idx
                    neuron_spike_times = spike_t[neuron_spike_mask]
                    spike_time_mask = (neuron_spike_times >= time_window[0]) & (neuron_spike_times <= time_window[1])
                    spike_times = neuron_spike_times[spike_time_mask] / ms
                
                # Get threshold for this neuron type
                threshold = -20  # default threshold
                if thresholds and name in thresholds:
                    threshold = thresholds[name]
                
                plt.figure(figsize=(12, 4)) 
                plt.plot(t[mask], v[mask], linewidth=1.5, color='#2E86AB')
                
                # Plot spikes at threshold level
                if len(spike_times) > 0:
                    spike_voltages = [threshold] * len(spike_times)
                    plt.scatter(spike_times, spike_voltages, color='red', s=12, 
                              marker='o', alpha=0.8, zorder=5)
                
                display_name = display_names.get(name, name) if display_names else name
                plt.title(f"{display_name} Membrane Potential Zoom ({time_window[0]/ms:.0f}-{time_window[1]/ms:.0f} ms)", 
                         fontsize=14, fontweight='bold')
                plt.xlabel("Time (ms)", fontsize=12)
                plt.ylabel("V (mV)", fontsize=12)
                if y_range:
                    plt.ylim(y_range)
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

            # Get spike times for this neuron if spike_monitors provided
            spike_times = []
            if spike_monitors and group_name in spike_monitors:
                spike_monitor = spike_monitors[group_name]
                spike_t, spike_i = get_monitor_spikes(spike_monitor)
                neuron_spike_mask = spike_i == neuron_idx
                neuron_spike_times = spike_t[neuron_spike_mask]
                spike_time_mask = (neuron_spike_times >= time_window[0]) & (neuron_spike_times <= time_window[1])
                spike_times = neuron_spike_times[spike_time_mask] / ms
            
            # Get threshold for this neuron type
            threshold = -20  # default threshold
            if thresholds and group_name in thresholds:
                threshold = thresholds[group_name]

            plt.figure(figsize=(12, 4)) 
            plt.plot(t[mask], v[mask], linewidth=1.5, color='#2E86AB')
            
            # Plot spikes at threshold level
            if len(spike_times) > 0:
                spike_voltages = [threshold] * len(spike_times)
                plt.scatter(spike_times, spike_voltages, color='red', s=12, 
                          marker='o', alpha=0.8, zorder=5)
            
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