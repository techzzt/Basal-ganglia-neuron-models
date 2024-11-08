import matplotlib.pyplot as plt
import os
from brian2 import *
from datetime import datetime

def plot_raster(spike_monitors):

    try:
        n_plots = len(spike_monitors)
        
        fig, axes = plt.subplots(n_plots, 1, figsize=(12, 3*n_plots))
        if n_plots == 1:
            axes = [axes]
        
        # Neuron spike plotting 
        for i, (name, monitor) in enumerate(spike_monitors.items()):
            axes[i].scatter(monitor.t/ms, monitor.i, s=1)
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