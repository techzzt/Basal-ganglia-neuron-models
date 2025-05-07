from brian2 import *
import numpy as np 

def compute_sta(pre_monitors, post_monitors, neuron_groups, synapses, window=100*ms, start_from_end=5000*ms):
    sta_results = {}
    t_end = defaultclock.t
    t_start = t_end - start_from_end

    for post_name, post_mon in post_monitors.items():
        print(f"\n[STA] Post: {post_name}")
        post_spike_times = post_mon.t
        post_spike_indices = post_mon.i

        valid_indices = np.where(post_spike_times >= t_start)[0]
        post_spike_times = post_spike_times[valid_indices]
        post_spike_indices = post_spike_indices[valid_indices]

        if len(post_spike_times) == 0:
            print(f"No spikes in last {start_from_end/ms} ms for {post_name}")
            continue

        sta_results[post_name] = {}

        for pre_name, pre_mon in pre_monitors.items():
            if pre_name == post_name:
                continue

            relevant_syns = [s for s in synapses if s.source.name == pre_name and s.target.name == post_name]
            if not relevant_syns:
                continue

            print(f"  Pre: {pre_name} — {len(post_spike_times)} spikes")
            pre_spike_times = pre_mon.t

            sta = []
            for spike_time in post_spike_times:
                idx = np.where((pre_spike_times >= spike_time - window) & (pre_spike_times < spike_time))[0]
                sta.append(len(idx))

            sta_results[post_name][pre_name] = np.mean(sta)

    return sta_results
