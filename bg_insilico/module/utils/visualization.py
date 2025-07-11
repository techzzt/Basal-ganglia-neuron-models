import matplotlib
import matplotlib.pyplot as plt
import os
import numpy as np 
from brian2 import *

# Improved backend setup for graph display on macOS
try:
    matplotlib.use('TkAgg')
except:
    try:
        matplotlib.use('Qt5Agg')
    except:
        matplotlib.use('Agg')  # Use non-interactive mode if backend not available

# Graph display settings
# plt.ion()  # Disable interactive mode to prevent graphs from disappearing 

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

def plot_raster(spike_monitors, sample_size=30, plot_order=None, start_time=0*ms, end_time=1000*ms, display_names=None, save_plot=True):
    np.random.seed(2025)
    try:
        if plot_order:
            spike_monitors = {name: spike_monitors[name] for name in plot_order if name in spike_monitors}

        if not spike_monitors:
            print("No valid neuron groups to plot.")
            return
        
        n_plots = len(spike_monitors)
        fig, axes = plt.subplots(n_plots, 1, figsize=(12, 2 * n_plots), sharex=True)
        if n_plots == 1:
            axes = [axes]

        firing_rates = {}
        
        print(f"\nRaster plot Range: {start_time/ms:.0f}ms - {end_time/ms:.0f}ms")
          
        for i, (name, monitor) in enumerate(spike_monitors.items()):
            spike_times, spike_indices = get_monitor_spikes(monitor)
            
            if len(spike_indices) == 0:
                print(f"No spikes recorded for {name}")
                continue

            total_neurons = monitor.source.N
            chosen_neurons = np.random.choice(total_neurons, size=min(sample_size, total_neurons), replace=False)

            time_mask = (spike_times >= start_time) & (spike_times <= end_time)
            neuron_mask = np.isin(spike_indices, chosen_neurons)
            combined_mask = time_mask & neuron_mask

            display_t = spike_times[combined_mask]
            display_i = spike_indices[combined_mask]

            display_name = display_names.get(name, name) if display_names else name
            axes[i].scatter(display_t / ms, display_i, s=0.1)
            axes[i].set_title(f'{display_name} Raster Plot (subset of {len(chosen_neurons)} neurons)')
            axes[i].set_ylabel('Neuron index')

            axes[i].set_ylim(min(chosen_neurons) - 1, max(chosen_neurons) + 1)
            axes[i].set_xlim(int(start_time/ms), int(end_time/ms))
            
            print(f"{display_name}: {len(display_t)} spikes shown (sampled from {sample_size} neurons)")

        plt.xlabel('Time (ms)')
        plt.tight_layout()
        
        # Save the plot to file for permanent viewing
        if save_plot:
            filename = 'raster_plot.png'
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"Raster plot saved to '{filename}'")
        
        try:
            plt.show(block=False)
            plt.pause(3.0)  # Show for 3 seconds
            print("Raster plot displayed. Plot saved to file for permanent viewing.")
        except Exception as e:
            print(f"Error displaying raster plot: {e}")
        finally:
            # Don't close immediately, let it stay open
            pass  

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
    
    try:
        plt.show(block=False)  
        plt.pause(0.1)  
    except Exception as e:
        print(f"Error displaying membrane potential plot: {e}")
        plt.close()  

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
            spike_times, spike_indices = get_monitor_spikes(monitor)
            
            if len(spike_indices) == 0:
                print(f"No spikes recorded for {name}")
                continue

            neuron_mask = spike_indices == neuron_index
            neuron_spike_times = spike_times[neuron_mask]

            if len(neuron_spike_times) == 0:
                print(f"{name} - Neuron {neuron_index} has no spikes.")
                continue

            axes[i].scatter(neuron_spike_times/ms, np.full_like(neuron_spike_times/ms, neuron_index), s=0.5, color='red')
            axes[i].set_title(f'{name} Neuron {neuron_index} Raster')
            axes[i].set_ylabel('Neuron index')
            
            if len(spike_times) > 0:
                axes[i].set_xlim(0, int(spike_times[-1] / ms))
            axes[i].set_ylim(neuron_index - 1, neuron_index + 1)

        plt.xlabel('Time (ms)')
        plt.tight_layout()
        
        try:
            plt.show(block=False)  
            plt.pause(0.1) 
        except Exception as e:
            print(f"Error displaying single neuron raster plot: {e}")
            plt.close() 

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
            spike_times, spike_indices = get_monitor_spikes(monitor)
            
            if len(spike_indices) == 0:
                print(f"No spikes recorded for {name}")
                continue

            time_mask = (spike_times >= stim_start) & (spike_times <= end_time)
            display_t = spike_times[time_mask]
            display_i = spike_indices[time_mask]

            axes[i].scatter(display_t / ms, display_i, s=0.5, color='darkblue')
            axes[i].set_title(f'{name} Raster (All neurons, {int(stim_start/ms)}–{int(end_time/ms)} ms)')
            axes[i].set_ylabel('Neuron index')
            axes[i].set_xlim(int(stim_start/ms), int(end_time/ms))
            axes[i].set_ylim(-1, monitor.source.N)

        plt.xlabel('Time (ms)')
        plt.tight_layout()
        
        try:
            plt.show(block=False)  
            plt.pause(0.1) 
        except Exception as e:
            print(f"Error displaying raster all neuron stim window: {e}")
            plt.close()

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
    
    try:
        plt.show(block=False)
        plt.pause(0.1)  
    except Exception as e:
        print(f"Error displaying Isyn plot: {e}")
        plt.close()  

