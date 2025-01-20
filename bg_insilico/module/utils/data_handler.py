import matplotlib.pyplot as plt
import os
import numpy as np 
from brian2 import *
from datetime import datetime

def plot_raster(spike_monitors, sample_size=30):  
    try:
        filtered_monitors = {name: monitor for name, monitor in spike_monitors.items() 
                           if not name.lower().startswith('cortex_')}
        
        n_plots = len(filtered_monitors)
        
        fig, axes = plt.subplots(n_plots, 1, figsize=(12, 3*n_plots))
        if n_plots == 1:
            axes = [axes]
        
        # Neuron spike plotting with random sampling
        for i, (name, monitor) in enumerate(filtered_monitors.items()):
            unique_neurons = np.unique(monitor.i)
            actual_sample_size = min(sample_size, len(unique_neurons))
            sampled_neurons = np.random.choice(unique_neurons, size=actual_sample_size, replace=False)
            
            mask = np.isin(monitor.i, sampled_neurons)
            sampled_times = monitor.t[mask]
            sampled_indices = monitor.i[mask]
            
            index_map = {old: new for new, old in enumerate(sorted(sampled_neurons))}
            mapped_indices = np.array([index_map[idx] for idx in sampled_indices])
            
            axes[i].scatter(sampled_times/ms, mapped_indices, s = 0.7)
            axes[i].set_title(f'{name} Raster Plot')
            axes[i].set_ylabel('Neuron index')
            axes[i].set_xlim(0, 1000)

        axes[-1].set_xlabel('Time (ms)')
        plt.subplots_adjust(hspace=0.5)
        
        # set directory 
        save_dir = 'results/raster_plots'
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(save_dir, f'raster_plot_{timestamp}.png')

        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"Raster plot saved: {filename}")
        
        plt.show()
        
    except Exception as e:
        print(f"Raster plot Error: {str(e)}")