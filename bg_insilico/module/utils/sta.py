from brian2 import *
import numpy as np 
import time

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

seed(int(time.time()))  
print(f"Using random seed: {int(time.time())}")

def compute_sta(pre_monitors, post_monitors, neuron_groups, synapses, connections, window=100*ms, bin_size=10*ms, start_from_end=5000*ms, min_spikes=10):
    
    sta_results = {}
    t_end = defaultclock.t
    t_start = t_end - start_from_end
    n_bins = int(window/bin_size)
    bins = np.linspace(-window/ms, 0, n_bins+1)

    print(f"\n== Spike Triggered Histogram (Window={window/ms} ms, Bin={bin_size/ms} ms, Last {start_from_end/ms} ms) ==")

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

            print(f" ← {pre_name:10s}: {hist.sum()} pre-spikes in window")

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
    print(f"Time window: {start_time/ms:.0f}ms - {end_time/ms:.0f}ms")
    
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
        
        print(f"Time window spikes: {total_spikes_in_window}")
        print(f"Number of activate neurons: {unique_active_neurons}")

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
