import matplotlib.pyplot as plt
import os
import numpy as np 
from brian2 import *
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np

def plot_raster(spike_monitors, sample_size=30, plot_order=None):
    try:
        if plot_order:
            spike_monitors = {name: spike_monitors[name] for name in plot_order if name in spike_monitors}

        if not spike_monitors:
            print("No valid neuron groups to plot.")
            return
        
        n_plots = len(spike_monitors)
        fig, axes = plt.subplots(n_plots, 1, figsize=(12, 3 * n_plots), sharex=True)

        if n_plots == 1:
            axes = [axes]
        
        firing_rates = {}  

        for i, (name, monitor) in enumerate(spike_monitors.items()):
            if len(monitor.i) == 0:
                print(f"No spikes recorded for {name}")
                continue

            unique_neurons = np.unique(monitor.i)
            actual_sample_size = min(sample_size, len(unique_neurons))
            sampled_neurons = np.random.choice(unique_neurons, size=actual_sample_size, replace=False)

            mask = np.isin(monitor.i, sampled_neurons)
            sampled_times = monitor.t[mask]
            sampled_indices = monitor.i[mask]

            index_map = {old: new for new, old in enumerate(sorted(sampled_neurons))}
            mapped_indices = np.array([index_map[idx] for idx in sampled_indices])

            axes[i].scatter(sampled_times/ms, mapped_indices, s=0.7)
            axes[i].set_title(f'{name} Raster Plot')
            axes[i].set_ylabel('Neuron index')
            axes[i].set_xlim(0, 1000)

            # 🔹 올바른 발화율 계산 (전체 뉴런 수 기준)
            num_spikes = len(monitor.t)
            total_neurons = monitor.source.N  # 전체 뉴런 개수
            simulation_time_sec = monitor.t[-1] / second  # 시뮬레이션 시간 (초)

            firing_rate = num_spikes / (total_neurons * simulation_time_sec) if total_neurons > 0 else 0  # Hz
            firing_rates[name] = firing_rate

            print(f"{name} 평균 발화율: {firing_rate:.2f} Hz")

            return firing_rates 

    except Exception as e:
        print(f"Raster plot Error: {str(e)}")


def plot_membrane_potential(voltage_monitors, plot_order=None):
    plt.figure(figsize=(10, 5))
    
    if plot_order:
        filtered_monitors = {name: monitor for name, monitor in voltage_monitors.items() if name in plot_order}
    else:
        filtered_monitors = voltage_monitors

    for name, monitor in filtered_monitors.items():
        if len(monitor.v) == 0:
            print(f"Warning: No voltage data recorded for {name}")
            continue
        plt.plot(monitor.t / ms, monitor.v[0] / mV, label=f'{name} Neuron 0')
    
    plt.xlabel('Time (ms)')
    plt.ylabel('Membrane Potential (mV)')
    plt.title('Membrane Potential Over Time')
    plt.legend()
    plt.show()
