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
        fig, axes = plt.subplots(n_plots, 1, figsize=(10, 2 * n_plots), sharex=True)

        if n_plots == 1:
            axes = [axes]
        
        firing_rates = {}

        for i, (name, monitor) in enumerate(spike_monitors.items()):
            if len(monitor.i) == 0:
                print(f"No spikes recorded for {name}")
                continue

            axes[i].scatter(monitor.t/ms, monitor.i, s=0.7)
            axes[i].set_title(f'{name} Raster Plot')
            axes[i].set_ylabel('Neuron index')

            num_spikes = len(monitor.t)
            total_neurons = monitor.source.N
            simulation_time_sec = monitor.t[-1] / second  

            firing_rate = num_spikes / (total_neurons * (defaultclock.t / second))
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
    
    plt.ylabel('Membrane Potential (mV)')
    plt.title('Membrane Potential Over Time')
    plt.legend()
    plt.show()
