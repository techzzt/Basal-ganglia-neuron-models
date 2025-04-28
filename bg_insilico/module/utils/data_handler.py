import matplotlib.pyplot as plt
import os
import numpy as np 
from brian2 import *
from datetime import datetime

def plot_raster(spike_monitors, sample_size=30, plot_order=None, start_time=0*ms, end_time=10000*ms):
    np.random.seed(2025)
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
          
        for i, (name, monitor) in enumerate(spike_monitors.items()):
            if len(monitor.i) == 0:
                print(f"No spikes recorded for {name}")
                continue

            total_neurons = monitor.source.N
            chosen_neurons = np.random.choice(total_neurons, size=min(sample_size, total_neurons), replace=False)

            time_mask = (monitor.t >= start_time) & (monitor.t <= end_time)
            neuron_mask = np.isin(monitor.i, chosen_neurons)
            combined_mask = time_mask & neuron_mask

            display_t = monitor.t[combined_mask]
            display_i = monitor.i[combined_mask]

            spike_count = len(display_t)
            time_window_sec = (end_time - 2000 * ms) / second
            firing_rate = spike_count / (len(chosen_neurons) * time_window_sec)
            firing_rates[name] = firing_rate

            axes[i].scatter(display_t / ms, display_i, s=0.7)
            axes[i].set_title(f'{name} Raster Plot (subset of {len(chosen_neurons)} neurons)')
            axes[i].set_ylabel('Neuron index')

            axes[i].set_ylim(min(chosen_neurons) - 1, max(chosen_neurons) + 1)
            axes[i].set_xlim(int(start_time/ms), int(end_time/ms))

            print(f"{name} 평균 발화율 ({int(2000 * ms/ms)}–{int(end_time/ms)}ms, {len(chosen_neurons)} neurons): {firing_rate:.2f} Hz")

        plt.xlabel('Time (ms)')
        plt.tight_layout()
        plt.show()

        return firing_rates

    except Exception as e:
        print(f"Raster plot Error: {str(e)}")

def plot_membrane_potential(voltage_monitors, plot_order=None):
    if plot_order:
        filtered_monitors = {name: voltage_monitors[name] for name in plot_order if name in voltage_monitors}
    else:
        filtered_monitors = voltage_monitors

    n_plots = len(filtered_monitors)
    if n_plots == 0:
        print("No voltage monitors to plot.")
        return

    fig, axes = plt.subplots(n_plots, 1, figsize=(10, 2.5 * n_plots), sharex=True)

    if n_plots == 1:
        axes = [axes]  

    for i, (name, monitor) in enumerate(filtered_monitors.items()):
        if len(monitor.v) == 0:
            print(f"Warning: No voltage data recorded for {name}")
            continue
        axes[i].plot(monitor.t / ms, monitor.v[0] / mV)
        axes[i].set_title(f'{name} Neuron 0 Membrane Potential')
        axes[i].set_ylabel('V (mV)')
        axes[i].set_xlim(0, int(monitor.t[-1] / ms))
    
    axes[-1].set_xlabel('Time (ms)')
    plt.tight_layout()
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


def plot_raster_all_neurons_stim_window(spike_monitors, stim_start=200*ms, end_time=1000*ms, plot_order=None):
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

            time_mask = (monitor.t >= stim_start) & (monitor.t <= end_time)
            display_t = monitor.t[time_mask]
            display_i = monitor.i[time_mask]

            axes[i].scatter(display_t / ms, display_i, s=0.5, color='darkblue')
            axes[i].set_title(f'{name} Raster (All neurons, {int(stim_start/ms)}–{int(end_time/ms)} ms)')
            axes[i].set_ylabel('Neuron index')
            axes[i].set_xlim(int(stim_start/ms), int(end_time/ms))
            axes[i].set_ylim(-1, monitor.source.N)

        plt.xlabel('Time (ms)')
        plt.tight_layout()
        plt.show()

    except Exception as e:
        print(f"Raster all neuron stim window error: {str(e)}")


def plot_isyn(voltage_monitors, plot_order=None):
    plt.figure(figsize=(10, 6))

    if plot_order:
        filtered_monitors = {name: voltage_monitors[name] for name in plot_order if name in voltage_monitors}
    else:
        filtered_monitors = voltage_monitors

    for name, monitor in filtered_monitors.items():
        if hasattr(monitor, 'Isyn'):
            plt.plot(monitor.t / ms, monitor.Isyn[0] / pA, label=f'{name} Isyn')
    
    plt.title('Synaptic Current (Isyn) Over Time')
    plt.xlabel('Time (ms)')
    plt.ylabel('Isyn (pA)')
    plt.legend()
    plt.tight_layout()
    plt.show()