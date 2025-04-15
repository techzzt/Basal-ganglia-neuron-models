import matplotlib.pyplot as plt
import os
import numpy as np 
from brian2 import *
from datetime import datetime

def plot_raster(spike_monitors, sample_size=30, plot_order=None):
    try:
        if plot_order:
            spike_monitors = {name: spike_monitors[name] for name in plot_order if name in spike_monitors}

        if not spike_monitors:
            print("No valid neuron groups to plot.")
            return
        
        n_plots = len(spike_monitors)
        fig, axes = plt.subplots(n_plots, 1, figsize=(10, 2 * n_plots), sharex=True)
        if n_plots == 1:
            axes = [axes]

        firing_rates = {}

        start_time = 1500 * ms
        end_time = 5000 * ms

        for i, (name, monitor) in enumerate(spike_monitors.items()):
            if len(monitor.i) == 0:
                print(f"No spikes recorded for {name}")
                continue

            total_neurons = monitor.source.N
            chosen_neurons = np.random.choice(total_neurons, size=min(sample_size, total_neurons), replace=False)

            neuron_mask = np.isin(monitor.i, chosen_neurons)
            display_t = monitor.t[neuron_mask]
            display_i = monitor.i[neuron_mask]

            valid_indices = (monitor.t >= start_time) & (monitor.t <= end_time) & neuron_mask
            valid_t = monitor.t[valid_indices]
            spike_count = len(valid_t)

            time_window_sec = (end_time - start_time) / second
            firing_rate = spike_count / (len(chosen_neurons) * time_window_sec)
            firing_rates[name] = firing_rate

            axes[i].scatter(display_t / ms, display_i, s=0.7)
            axes[i].set_title(f'{name} Raster Plot (subset of {len(chosen_neurons)} neurons)')
            axes[i].set_ylabel('Neuron index')

            axes[i].set_ylim(min(chosen_neurons) - 1, max(chosen_neurons) + 1)

            axes[i].set_xlim(0, int(monitor.t[-1] / ms))

            print(f"{name} 평균 발화율 ({int(start_time/ms)}–{int(end_time/ms)}ms, {len(chosen_neurons)} neurons): {firing_rate:.2f} Hz")

        plt.xlabel('Time (ms)')
        plt.tight_layout()
        plt.show()

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
    
    plt.ylabel('Membrane Potential (mV)')
    plt.title('Membrane Potential Over Time')
    plt.legend()
    plt.show()

# track individual neuron
def plot_single_neuron_raster(spike_monitors, neuron_index, plot_order=None):
    try:
        if plot_order:
            spike_monitors = {name: spike_monitors[name] for name in plot_order if name in spike_monitors}

        if not spike_monitors:
            print("No valid neuron groups to plot.")
            return

        n_plots = len(spike_monitors)
        fig, axes = plt.subplots(n_plots, 1, figsize=(10, 2 * n_plots), sharex=True)
        if n_plots == 1:
            axes = [axes]

        for i, (name, monitor) in enumerate(spike_monitors.items()):
            if len(monitor.i) == 0:
                print(f"No spikes recorded for {name}")
                continue

            # 해당 neuron index의 spike만 필터링
            neuron_mask = monitor.i == neuron_index
            spike_times = monitor.t[neuron_mask]

            if len(spike_times) == 0:
                print(f"{name} - Neuron {neuron_index} has no spikes.")
                continue

            axes[i].scatter(spike_times/ms, np.full_like(spike_times/ms, neuron_index), s=4, color='red')
            axes[i].set_title(f'{name} Neuron {neuron_index} Raster')
            axes[i].set_ylabel('Neuron index')
            axes[i].set_xlim(0, int(monitor.t[-1] / ms))
            axes[i].set_ylim(neuron_index - 1, neuron_index + 1)

        plt.xlabel('Time (ms)')
        plt.tight_layout()
        plt.show()

    except Exception as e:
        print(f"Single neuron raster plot Error: {str(e)}")
