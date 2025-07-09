from brian2 import *
import numpy as np
import time
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

plt.ion()  

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

def plot_isi_histogram(isi_list, neuron_name, bins=30, save_plot=True, display_name=None):

    if not isi_list:
        print(f"[{neuron_name}] No ISI data available.")
        return
    
    # Apply display_name
    if display_name is None:
        display_name = neuron_name
    
    plt.figure(figsize=(10, 6))
    
    counts, bin_edges, _ = plt.hist(isi_list, bins=bins, alpha=0.7, color='skyblue', edgecolor='black')
    
    mean_isi = np.mean(isi_list)
    std_isi = np.std(isi_list)
    cv_isi = std_isi / mean_isi if mean_isi > 0 else 0
    
    plt.axvline(mean_isi, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_isi:.1f} ms')
    plt.axvline(mean_isi + std_isi, color='orange', linestyle=':', linewidth=1, label=f'+1 SD: {mean_isi + std_isi:.1f} ms')
    plt.axvline(mean_isi - std_isi, color='orange', linestyle=':', linewidth=1, label=f'-1 SD: {mean_isi - std_isi:.1f} ms')
    
    plt.xlabel('ISI Interval (ms)')
    plt.ylabel('Counts/bin')
    plt.title(f'ISI Distribution - {display_name}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_plot:
        filename = f'isi_histogram_{neuron_name}.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"ISI histogram saved to '{filename}'")
    
    try:
        plt.show(block=False) 
        plt.pause(0.1)  
    except Exception as e:
        print(f"Error displaying graph: {e}")
        plt.close()  

def plot_all_isi_histograms(isi_results, bins=30, save_plot=True, display_names=None):

    if not isi_results:
        print("No ISI results to plot")
        return
    
    try:
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()
        
        neuron_names = list(isi_results.keys())
        
        for i, (neuron_name, result) in enumerate(isi_results.items()):
            if i >= 6:
                break
                
            ax = axes[i]
            isi_list = result.get('isi_list', [])
            
            # display_name 적용
            display_name = display_names.get(neuron_name, neuron_name) if display_names else neuron_name
            
            if not isi_list or len(isi_list) == 0:
                ax.text(0.5, 0.5, f'{display_name}\nNo ISI data', 
                       ha='center', va='center', transform=ax.transAxes)
                ax.set_title(f'{display_name}')
                continue
            
            try:
                counts, bin_edges, _ = ax.hist(isi_list, bins=bins, alpha=0.7, color='skyblue', edgecolor='black')
                
                mean_isi = np.mean(isi_list)
                std_isi = np.std(isi_list)
                cv_isi = std_isi / mean_isi if mean_isi > 0 else 0
                
                ax.axvline(mean_isi, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_isi:.1f} ms')
                
                ax.set_xlabel('ISI Interval (ms)')
                ax.set_ylabel('Counts/bin')
                ax.set_title(f'{display_name}')
                ax.grid(True, alpha=0.3)
                
                stats_text = f'Mean: {mean_isi:.1f} ms\nStd: {std_isi:.1f} ms\nCV: {cv_isi:.3f}\nN: {len(isi_list)}'
                ax.text(0.05, 0.95, stats_text, transform=ax.transAxes, 
                       verticalalignment='top', horizontalalignment='left',
                       fontsize=9, color='black', 
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
                       
            except Exception as e:
                print(f"Error plotting histogram for {neuron_name}: {e}")
                ax.text(0.5, 0.5, f'{display_name}\nError plotting', 
                       ha='center', va='center', transform=ax.transAxes)
                ax.set_title(f'{display_name}')
        
        for i in range(len(neuron_names), 6):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        
        if save_plot:
            filename = 'isi_histograms_all_neurons.png'
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"모든 뉴런의 ISI 히스토그램이 '{filename}'에 저장되었습니다.")
        
        try:
            plt.show(block=False)
            plt.pause(0.1)
        except Exception as e:
            print(f"Error displaying ISI histogram: {e}")
            plt.close()
            
    except Exception as e:
        print(f"Error in plot_all_isi_histograms: {e}")
        import traceback
        traceback.print_exc()

def calculate_isi_in_chunks(spike_times, spike_indices, chunk_size=10000, max_total_samples=100000):

    all_isi = []
    neuron_counts = {}
    
    unique_neurons = np.unique(spike_indices)
    total_neurons = len(unique_neurons)
    
    # Process neurons in chunks
    for chunk_start in range(0, total_neurons, chunk_size):
        chunk_end = min(chunk_start + chunk_size, total_neurons)
        chunk_neurons = unique_neurons[chunk_start:chunk_end]
        
        for neuron_idx in chunk_neurons:
            neuron_spikes = spike_times[spike_indices == neuron_idx]
            if len(neuron_spikes) > 1:
                isi = np.diff(neuron_spikes)
                isi_ms = isi / ms
                
                # Check if adding these ISI would exceed our limit
                if len(all_isi) + len(isi_ms) > max_total_samples:
                    remaining_slots = max_total_samples - len(all_isi)
                    if remaining_slots > 0:
                        # Randomly sample ISI values
                        if len(isi_ms) > remaining_slots:
                            sampled_indices = np.random.choice(len(isi_ms), remaining_slots, replace=False)
                            all_isi.extend(isi_ms[sampled_indices])
                        else:
                            all_isi.extend(isi_ms)
                    break  
                else:
                    all_isi.extend(isi_ms)
                
                neuron_counts[neuron_idx] = len(isi_ms)
        
        if len(all_isi) >= max_total_samples:
            break
    
    return all_isi, neuron_counts

def compute_isi_all_neurons(spike_monitors, start_time=0*ms, end_time=10000*ms, plot_order=None, return_dict=True, plot_histograms=False, display_names=None, x_axis_limits=None, y_axis_limits=None):

    isi_results = {}
    
    for name, monitor in spike_monitors.items():
        if plot_order and name not in plot_order:
            continue
            
        spike_times, spike_indices = get_monitor_spikes(monitor)
        total_neurons = monitor.source.N
        
        print(f"\n[{name}]")
        
        if len(spike_indices) == 0:
            isi_results[name] = {
                'mean_isi': 0.0,
                'std_isi': 0.0,
                'cv_isi': 0.0,
                'total_spikes': 0,
                'active_neurons': 0
            }
            print(f"No spikes")
            continue

        time_mask = (spike_times >= start_time) & (spike_times <= end_time)
        spike_times_filtered = spike_times[time_mask]
        neuron_ids_filtered = spike_indices[time_mask]

        total_spikes_in_window = len(spike_times_filtered)
        unique_active_neurons = len(np.unique(neuron_ids_filtered)) if len(neuron_ids_filtered) > 0 else 0
        
        if total_spikes_in_window < 2:
            isi_results[name] = {
                'mean_isi': 0.0,
                'std_isi': 0.0,
                'cv_isi': 0.0,
                'total_spikes': total_spikes_in_window,
                'active_neurons': unique_active_neurons
            }
            print(f"Not enough spikes for ISI calculation ({total_spikes_in_window} spikes)")
            continue
        
        all_isi = []
        for neuron_idx in np.unique(neuron_ids_filtered):
            neuron_spikes = spike_times_filtered[neuron_ids_filtered == neuron_idx]
            if len(neuron_spikes) > 1:
                isi = np.diff(neuron_spikes)
                isi_ms = isi / ms  
                all_isi.extend(isi_ms)
        
        if not all_isi:
            isi_results[name] = {
                'mean_isi': 0.0,
                'std_isi': 0.0,
                'cv_isi': 0.0,
                'total_spikes': total_spikes_in_window,
                'active_neurons': unique_active_neurons
            }
            print(f"No valid ISI intervals")
            continue
        
        mean_isi = np.mean(all_isi)
        std_isi = np.std(all_isi)
        cv_isi = std_isi / mean_isi if mean_isi > 0 else 0
        min_isi = np.min(all_isi)
        max_isi = np.max(all_isi)
        
        isi_results[name] = {
            'mean_isi': mean_isi,
            'std_isi': std_isi,
            'cv_isi': cv_isi,
            'min_isi': min_isi,
            'max_isi': max_isi,
            'total_spikes': total_spikes_in_window,
            'active_neurons': unique_active_neurons,
            'isi_count': len(all_isi)
        }
        
        print(f" ISI Analysis: {len(all_isi)} intervals from {unique_active_neurons} neurons")
        print(f"Mean ISI: {mean_isi:.2f} ± {std_isi:.2f} ms")
        print(f"ISI Range: {min_isi:.1f} - {max_isi:.1f} ms")
        print(f"ISI CV: {cv_isi:.3f}")
        
        isi_results[name]['isi_list'] = all_isi

    if plot_histograms and isi_results:
        print("\nCreating combined ISI histogram for all neurons...")
        
        plot_results = {}
        for name, result in isi_results.items():
            if 'isi_list' in result and result['isi_list']:
                plot_results[name] = result
            else:
                plot_results[name] = {
                    **result,
                    'isi_list': []
                }
        
        if plot_results:
            plot_all_isi_histograms(plot_results, bins=30, save_plot=True, display_names=display_names)
        else:
            print("No ISI data available for histogram plotting")

    if return_dict:
        return isi_results
    else:
        return list(isi_results.values())

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

def print_spike_pattern_comparison(comparison_results, label1="Condition 1", label2="Condition 2"):

    print(f"\n=== Spike 패턴 비교: {label1} vs {label2} ===")
    
    for neuron_name, result in comparison_results.items():
        print(f"\n--- {neuron_name} ---")
        
        result1 = result[label1]
        result2 = result[label2]
        
        print(f"{label1}:")
        print(f"  평균 ISI: {result1['mean_isi']:.2f} ms")
        print(f"  ISI 표준편차: {result1['std_isi']:.2f} ms")
        print(f"  ISI 변동계수: {result1['cv_isi']:.3f}")
        print(f"  총 spike 수: {result1['total_spikes']}")
        
        print(f"{label2}:")
        print(f"  평균 ISI: {result2['mean_isi']:.2f} ms")
        print(f"  ISI 표준편차: {result2['std_isi']:.2f} ms")
        print(f"  ISI 변동계수: {result2['cv_isi']:.3f}")
        print(f"  총 spike 수: {result2['total_spikes']}")
        
        if 'changes' in result:
            changes = result['changes']
            print(f"변화량:")
            print(f"  평균 ISI 변화: {changes['isi_change_percent']:+.1f}%")
            print(f"  ISI 변동계수 변화: {changes['cv_change']:+.3f}")
            print(f"  총 spike 수 변화: {changes['spike_change_percent']:+.1f}%")

def analyze_isi_distribution(isi_list, bins=30):

    if not isi_list:
        return None
    
    isi_array = np.array(isi_list)

    mean_isi = np.mean(isi_array)
    median_isi = np.median(isi_array)
    std_isi = np.std(isi_array)
    
    hist, bin_edges = np.histogram(isi_array, bins=bins)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    mode_idx = np.argmax(hist)
    mode_isi = bin_centers[mode_idx]
    
    skewness = np.mean(((isi_array - mean_isi) / std_isi) ** 3) if std_isi > 0 else 0
    
    return {
        'mean_isi': mean_isi,
        'median_isi': median_isi,
        'mode_isi': mode_isi,
        'std_isi': std_isi,
        'skewness': skewness,
        'histogram': {
            'counts': hist,
            'bin_centers': bin_centers,
            'bin_edges': bin_edges
        }
    }

def compute_sta(pre_monitors, post_monitors, neuron_groups, synapses, connections, window=100*ms, bin_size=10*ms, start_from_end=5000*ms, min_spikes=10):
    
    sta_results = {}
    t_end = defaultclock.t
    t_start = t_end - start_from_end
    n_bins = int(window/bin_size)
    bins = np.linspace(-window/ms, 0, n_bins+1)

    connected_pairs = set((conn['pre'], conn['post']) for conn in connections.values())

    for post_name, post_mon in post_monitors.items():
        post_spike_times, _ = get_monitor_spikes(post_mon)
        valid_mask = (post_spike_times >= t_start)
        post_spike_times = post_spike_times[valid_mask]

        if len(post_spike_times) < min_spikes:
            print(f"[{post_name}] Not enough spikes ({len(post_spike_times)})")
            continue

        print(f"\n[Post: {post_name}] Spike count: {len(post_spike_times)}")
        sta_results[post_name] = {}

        for pre_name, pre_mon in pre_monitors.items():
            if pre_name == post_name:
                continue

            if (pre_name, post_name) not in connected_pairs:
                continue

            pre_spike_times, _ = get_monitor_spikes(pre_mon)
            all_deltas = []

            for t_post in post_spike_times:
                mask = (pre_spike_times >= t_post - window) & (pre_spike_times < t_post)
                deltas = (pre_spike_times[mask] - t_post) / ms 
                all_deltas.extend(deltas)

            hist, _ = np.histogram(all_deltas, bins=bins)
            sta_results[post_name][pre_name] = hist

    return sta_results, bins

def estimate_required_weight_adjustment(observed, target):

    weight_adjustments = {}
    for neuron, obs_rate in observed.items():
        tgt_rate = target.get(neuron)
        if obs_rate > 0 and tgt_rate is not None:
            weight_adjustments[neuron] = round(tgt_rate / obs_rate, 3)
        elif obs_rate == 0 and tgt_rate: 
            weight_adjustments[neuron] = 10.0  
        else:
            weight_adjustments[neuron] = None
    return weight_adjustments


def adjust_connection_weights(connections, weight_adjustments):

    updated_connections = {}

    for conn_name, conn in connections.items():
        post = conn.get('post')
        scale = weight_adjustments.get(post)

        if scale is not None:
            original_weight = conn.get('weight', 1.0)
            new_weight = round(original_weight * scale, 3)

            conn = conn.copy()
            conn['weight'] = new_weight

        updated_connections[conn_name] = conn

    return updated_connections

def compute_firing_rates_all_neurons(spike_monitors, start_time=0*ms, end_time=10000*ms, plot_order=None, return_dict=True):
    
    firing_rates = {}
    
    for name, monitor in spike_monitors.items():
        if plot_order and name not in plot_order:
            continue
            
        spike_times, spike_indices = get_monitor_spikes(monitor)
        total_neurons = monitor.source.N
        time_window_sec = (end_time - start_time) / second
        
        print(f"\n[{name}]")
        
        if len(spike_indices) == 0:
            firing_rates[name] = 0.0
            print(f"No spikes")
            continue

        time_mask = (spike_times >= start_time) & (spike_times <= end_time)
        spike_times_filtered = spike_times[time_mask]
        neuron_ids_filtered = spike_indices[time_mask]

        total_spikes_in_window = len(spike_times_filtered)
        unique_active_neurons = len(np.unique(neuron_ids_filtered)) if len(neuron_ids_filtered) > 0 else 0
        
        if total_neurons > 0 and time_window_sec > 0:
            network_avg_rate = total_spikes_in_window / (total_neurons * time_window_sec)
            firing_rates[name] = network_avg_rate
            print(f" Rates: {total_spikes_in_window} / ({total_neurons} × {time_window_sec:.1f}) = {network_avg_rate:.4f} Hz")
            print(f"Mean Firing Rates {network_avg_rate:.2f} Hz")
        else:
            firing_rates[name] = 0.0
            print(f"No spikes")

    if return_dict:
        return firing_rates
    else:
        return list(firing_rates.values())

def analyze_stimulus_effect(spike_monitors, stimulus_start=10000, stimulus_duration=1000, window=1000):

    print("\n=== Stimulus Effect Analysis ===")
    
    # Time windows
    pre_start = stimulus_start - window
    pre_end = stimulus_start
    stim_start = stimulus_start
    stim_end = stimulus_start + stimulus_duration
    post_start = stim_end
    post_end = stim_end + window
    
    for name, monitor in spike_monitors.items():
        if len(monitor.t) == 0:
            print(f"{name}: No spikes recorded")
            continue
            
        spike_times_ms = monitor.t / ms
        total_neurons = monitor.source.N
        
        pre_spikes = np.sum((spike_times_ms >= pre_start) & (spike_times_ms < pre_end))
        stim_spikes = np.sum((spike_times_ms >= stim_start) & (spike_times_ms < stim_end))
        post_spikes = np.sum((spike_times_ms >= post_start) & (spike_times_ms < post_end))
        
        pre_rate = pre_spikes / (window/1000) / total_neurons
        stim_rate = stim_spikes / (stimulus_duration/1000) / total_neurons
        post_rate = post_spikes / (window/1000) / total_neurons
        
        stim_change = ((stim_rate - pre_rate) / pre_rate * 100) if pre_rate > 0 else 0
        
        print(f"{name}:")
        print(f"  Pre-stimulus  ({pre_start}-{pre_end}ms): {pre_rate:.3f} Hz")
        print(f"  During stimulus ({stim_start}-{stim_end}ms): {stim_rate:.3f} Hz")
        print(f"  Post-stimulus ({post_start}-{post_end}ms): {post_rate:.3f} Hz")
        print(f"  Stimulus change: {stim_change:+.1f}%")
        print()
    
    return True

def debug_isi_data(spike_monitors, start_time=0*ms, end_time=10000*ms):

    print("\n=== ISI Data Debugging ===")
    
    for name, monitor in spike_monitors.items():
        print(f"\n[{name}]")
        
        print(f"  Monitor type: {type(monitor)}")
        print(f"  Total neurons: {monitor.source.N}")
        
        spike_times, spike_indices = get_monitor_spikes(monitor)
        print(f"  Total spikes: {len(spike_times)}")
        
        if len(spike_times) == 0:
            print(f"  Warning: No spikes!")
            continue
        
        time_mask = (spike_times >= start_time) & (spike_times <= end_time)
        spike_times_filtered = spike_times[time_mask]
        neuron_ids_filtered = spike_indices[time_mask]
        
        print(f"  Spikes in analysis window: {len(spike_times_filtered)}")
        print(f"  Active neurons: {len(np.unique(neuron_ids_filtered))}")
        
        all_isi = []
        for neuron_idx in np.unique(neuron_ids_filtered):
            neuron_spikes = spike_times_filtered[neuron_ids_filtered == neuron_idx]
            if len(neuron_spikes) > 1:
                isi = np.diff(neuron_spikes)
                isi_ms = isi / ms
                all_isi.extend(isi_ms)
        
        print(f"  Calculated ISI count: {len(all_isi)}")
        if all_isi:
            print(f"  ISI range: {np.min(all_isi):.1f} - {np.max(all_isi):.1f} ms")
            print(f"  Mean ISI: {np.mean(all_isi):.1f} ms")
        else:
            print(f"  Warning: No valid ISI intervals!")

def plot_all_isi_histograms_efficient(isi_results, bins=30, save_plot=True, display_names=None):

    if not isi_results:
        return
    
    global_x_min = float('inf')
    global_x_max = float('-inf')
    global_y_max = 0
    
    print("Calculating global ranges...")
    
    for neuron_name, result in isi_results.items():
        isi_list = result.get('isi_list', [])
        if not isi_list:
            continue
            
        counts, bin_edges, _ = plt.hist(isi_list, bins=bins, alpha=0)
        plt.close() 
        
        x_min = bin_edges[0]
        x_max = bin_edges[-1]
        y_max = np.max(counts)
        
        global_x_min = min(global_x_min, x_min)
        global_x_max = max(global_x_max, x_max)
        global_y_max = max(global_y_max, y_max)
    
    if global_x_min == float('inf'):
        global_x_min = 0
        global_x_max = 100
        global_y_max = 100
    
    x_margin = (global_x_max - global_x_min) * 0.1
    global_x_min = max(0, global_x_min - x_margin)
    global_x_max = global_x_max + x_margin
    global_y_max = global_y_max * 1.1
    
    print(f"Global ranges: X=[{global_x_min:.1f}, {global_x_max:.1f}], Y=[0, {global_y_max:.0f}]")
    
    fig, axes = plt.subplots(2, 3, figsize=(20, 14))
    axes = axes.flatten()
    
    neuron_names = list(isi_results.keys())
    
    for i, (neuron_name, result) in enumerate(isi_results.items()):
        if i >= 6:
            break
            
        ax = axes[i]
        isi_list = result.get('isi_list', [])
        
        display_name = display_names.get(neuron_name, neuron_name) if display_names else neuron_name
        
        if not isi_list or len(isi_list) == 0:
            ax.text(0.5, 0.5, f'{display_name}\nNo ISI data', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f'{display_name}')
            continue
        
        counts, bin_edges, _ = ax.hist(isi_list, bins=bins, alpha=0.7, color='skyblue', edgecolor='black')
        
        mean_isi = np.mean(isi_list)
        std_isi = np.std(isi_list)
        cv_isi = std_isi / mean_isi if mean_isi > 0 else 0
        
        ax.axvline(mean_isi, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_isi:.1f} ms')
        ax.axvline(mean_isi + std_isi, color='orange', linestyle=':', linewidth=1, label=f'+1 SD: {mean_isi + std_isi:.1f} ms')
        ax.axvline(mean_isi - std_isi, color='orange', linestyle=':', linewidth=1, label=f'-1 SD: {mean_isi - std_isi:.1f} ms')
        
        ax.set_xlabel('ISI Interval (ms)')
        ax.set_ylabel('Counts/bin')
        ax.set_title(f'{display_name}')
        ax.legend(fontsize=8, loc='upper right')
        ax.grid(True, alpha=0.3)
        
        ax.set_xlim(global_x_min, global_x_max)
        ax.set_ylim(0, global_y_max)
        
        # Place statistics tex in top-left to avoid overlap
        stats_text = f'Mean: {mean_isi:.1f} ms\nStd: {std_isi:.1f} ms\nCV: {cv_isi:.3f}\nN: {len(isi_list)}'
        ax.text(0.05, 0.95, stats_text, transform=ax.transAxes, 
               verticalalignment='top', horizontalalignment='left',
               fontsize=10, color='black', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    for i in range(len(neuron_names), 6):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    
    if save_plot:
        filename = 'isi_histograms_all_neurons_shared_axes.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"All neuron ISI histograms with shared axes saved to '{filename}'")
    
    try:
        plt.show(block=False)  
        plt.pause(0.1)  
    except Exception as e:
        print(f"Error displaying graph: {e}")
        plt.close()  
