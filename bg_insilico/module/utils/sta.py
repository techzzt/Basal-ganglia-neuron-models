from brian2 import *
import numpy as np 

def compute_sta(pre_monitors, post_monitors, neuron_groups, synapses, connections, window=100*ms, bin_size=10*ms, start_from_end=5000*ms, min_spikes=10):
    sta_results = {}
    t_end = defaultclock.t
    t_start = t_end - start_from_end
    n_bins = int(window/bin_size)
    bins = np.linspace(-window/ms, 0, n_bins+1)

    print(f"\n===== Spike-Triggered Histogram (Window={window/ms} ms, Bin={bin_size/ms} ms, Last {start_from_end/ms} ms) =====")

    connected_pairs = set((conn['pre'], conn['post']) for conn in connections.values())

    for post_name, post_mon in post_monitors.items():
        post_spike_times = post_mon.t
        valid_mask = (post_spike_times >= t_start)
        post_spike_times = post_spike_times[valid_mask]

        if len(post_spike_times) < min_spikes:
            print(f"[{post_name}] Not enough spikes ({len(post_spike_times)}) for reliable STA.")
            continue

        print(f"\n[Post: {post_name}] Spike count: {len(post_spike_times)}")
        sta_results[post_name] = {}

        for pre_name, pre_mon in pre_monitors.items():
            if pre_name == post_name:
                continue

            if (pre_name, post_name) not in connected_pairs:
                continue

            pre_spike_times = pre_mon.t
            all_deltas = []

            for t_post in post_spike_times:
                mask = (pre_spike_times >= t_post - window) & (pre_spike_times < t_post)
                deltas = (pre_spike_times[mask] - t_post) / ms 
                all_deltas.extend(deltas)

            hist, _ = np.histogram(all_deltas, bins=bins)
            sta_results[post_name][pre_name] = hist

            print(f"  ← {pre_name:10s}: {hist.sum()} pre-spikes in window")

    return sta_results, bins