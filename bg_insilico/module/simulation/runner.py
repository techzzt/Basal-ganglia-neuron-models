import numpy as np
import os
from copy import deepcopy
import time
from tqdm import tqdm
import math
import gc
import matplotlib.pyplot as plt

from brian2 import *
from brian2 import mV, ms, nS, Hz
from module.models.neuron_models import create_neurons
from module.models.Synapse import create_synapses
from module.utils.visualization import (
    plot_raster, plot_membrane_potential,
    plot_raster_all_neurons_stim_window
    )

from module.utils.sta import compute_firing_rates_all_neurons, adjust_connection_weights, estimate_required_weight_adjustment
from module.models.stimulus import create_poisson_inputs

# Matplotlib backend setup
# plt.ion() 

try:
    from brian2.devices.device import reset_device, get_device
    reset_device()
except:
    pass

import platform
if platform.system() == 'Darwin':
    os.environ['CC'] = '/usr/bin/clang'
    os.environ['CXX'] = '/usr/bin/clang++'

prefs.codegen.target = 'cython'
prefs.core.default_float_dtype = np.float32  
class SimulationMonitor:
    def __init__(self, total_time, dt=1*ms, update_interval=100*ms):  
        self.total_time = float(total_time/ms)  
        self.start_time = time.time()
        self.last_t = 0
        self.dt = float(dt/ms)
        self.update_interval = float(update_interval/ms)
        self.total_steps = int(self.total_time / self.dt)
        self.pbar = tqdm(total=self.total_time, unit='ms', desc='Simulation Progress')
        print(f"Total simulation time: {self.total_time} ms")
    
    def update(self, t):
        try:
            current_time = float(t/ms)
            if current_time - self.last_t >= self.update_interval: 
                progress = current_time - self.last_t
                self.pbar.update(progress)
                self.last_t = current_time
                
                elapsed = time.time() - self.start_time
                total_progress_fraction = current_time / self.total_time
                
                if total_progress_fraction > 0:
                    estimated_total = elapsed / total_progress_fraction
                    remaining = max(0, estimated_total - elapsed)
                    self.pbar.set_postfix(remaining=f"{remaining:.1f}s")
                
                if current_time % 1000 == 0: 
                    gc.collect()
        except Exception as e:
            print(f"Error updating progress: {str(e)}")
    
    def close(self):
        self.pbar.close()

def run_with_progress(net, duration, dt=10*ms, update_interval=500*ms):
    monitor = SimulationMonitor(duration, dt=dt, update_interval=update_interval)
    
    @network_operation(dt=update_interval)
    def update_progress(t):
        monitor.update(t)
    
    net.add(update_progress)
    
    try:
        net.run(duration, report='text')  
    finally:
        monitor.close()
        gc.collect()

def clear_monitor_history(spike_monitors, save_data=True, save_dir="./spike_data"):
    """
    Clear monitor history to save memory while optionally saving data
    """
    if save_data:
        os.makedirs(save_dir, exist_ok=True)
        timestamp = time.time()
        
        for name, monitor in spike_monitors.items():
            if monitor.num_spikes > 0:
                # Save spike data before clearing
                spike_data = {
                    'spike_times': np.array(monitor.t/ms),
                    'spike_indices': np.array(monitor.i),
                    'timestamp': timestamp
                }
                filename = f"{save_dir}/{name}_spikes_{int(timestamp)}.npz"
                np.savez_compressed(filename, **spike_data)
    
    # Clear monitor data (this doesn't affect neuron states)
    for monitor in spike_monitors.values():
        try:
            # Clear the monitor's internal arrays
            monitor.source._spikemonitor_indices = []
            monitor.source._spikemonitor_times = []
            if hasattr(monitor, 't'):
                monitor.t.resize(0)
            if hasattr(monitor, 'i'):
                monitor.i.resize(0)
        except:
            pass 
    
    gc.collect()
    print(f"Monitor history cleared, data saved to {save_dir}")

def get_monitor_spikes(monitor):
    """Helper function to get spike data from monitor, handling both regular and reconstructed monitors"""
    try:
        # Try to access regular monitor data first
        if hasattr(monitor, 't') and hasattr(monitor, 'i') and len(monitor.t) > 0:
            return monitor.t, monitor.i
        # Fall back to reconstructed data
        elif hasattr(monitor, '_spike_times') and hasattr(monitor, '_spike_indices'):
            return monitor._spike_times, monitor._spike_indices
        else:
            return np.array([]) * ms, np.array([])
    except:
        # Final fallback
        return np.array([]) * ms, np.array([])

def run_simulation_with_inh_ext_input(
    neuron_configs,
    connections,
    synapse_class,
    simulation_params,
    plot_order=None,
    start_time=0*ms,
    end_time=1000*ms,
    stim_pattern=None,
    ext_inputs=None,
    cleanup_interval=1000
):
    
    try:
        from brian2.devices.device import reset_device, get_device
        current_device = get_device()
        if hasattr(current_device, 'build') and hasattr(current_device, 'run'): 
            print("Reset device to runtime mode")
    except:
        pass
    
    spike_monitors = {}
    neuron_groups = None
    synapse_connections = None
    poisson_groups = None
    firing_rates = {}
    results = {'spike_monitors': {}, 'firing_rates': {}}
    
    try:
        start_scope()
        
        neuron_groups = create_neurons(neuron_configs, simulation_params, connections)
        
        synapse_connections = create_synapses(neuron_groups, connections, synapse_class)
        
        poisson_groups = create_poisson_inputs(neuron_groups, ext_inputs) if ext_inputs else []
        
        for name, group in neuron_groups.items():
            if not name.startswith(('Cortex_', 'Ext_')):  
                if group.N > 10000:  
                    sample_size = min(5, group.N) 
                elif group.N > 1000:
                    sample_size = min(10, group.N) 
                else:
                    sample_size = min(15, group.N) 
                sample_indices = np.random.choice(group.N, size=sample_size, replace=False).tolist()
                sp_mon = SpikeMonitor(group, record=sample_indices)
                spike_monitors[name] = sp_mon
        
        net = Network()
        net.add(*neuron_groups.values())
        net.add(*synapse_connections)
        net.add(*poisson_groups)
        net.add(*spike_monitors.values())
        
        duration = simulation_params.get('duration', 1000) * ms
        dt = simulation_params.get('dt', 10) * ms 
        print(f"\nStarting optimized simulation for {duration/ms} ms with memory management")
        print(f"History cleanup interval: {cleanup_interval} ms")
        print(f"Using timestep: {dt/ms} ms")
        defaultclock.dt = dt
        
        chunk_size = cleanup_interval * ms
        t = 0 * ms
        chunk_number = 0
        
        all_spike_data = {name: {'times': [], 'indices': []} for name in spike_monitors.keys()}
        
        while t < duration:
            run_time = min(chunk_size, duration - t)
            if chunk_number % 2 == 0:  
                print(f"Running chunk {chunk_number}: {t/ms} to {(t + run_time)/ms} ms")
            
            net.run(run_time, report=None)
            
            for name, monitor in spike_monitors.items():
                if monitor.num_spikes > 0:
                    absolute_times = np.array(monitor.t/ms) + (t/ms)
                    all_spike_data[name]['times'].extend(absolute_times)
                    all_spike_data[name]['indices'].extend(monitor.i)
            
            if chunk_number > 0:  
                if chunk_number % 2 == 0: 
                    print(f"Clearing monitor history at {(t + run_time)/ms} ms")
                for monitor in spike_monitors.values():
                    try:
                        monitor._resize(0)
                    except:
                        pass
                gc.collect()
            
            t += run_time
            chunk_number += 1
            
            if chunk_number % 3 == 0:  
                progress_pct = (t/duration)*100
                total_spikes = sum(len(data['times']) for data in all_spike_data.values())
                print(f"Progress: {progress_pct:.1f}% ({t/ms:.0f}/{duration/ms:.0f} ms), Total spikes: {total_spikes}")
        
        print("\nReconstructing final spike data...")
        for name, data in all_spike_data.items():
            monitor = spike_monitors[name]
            if len(data['times']) > 0:
                # Store in custom attributes to avoid read-only variable issues
                monitor._spike_times = np.array(data['times']) * ms
                monitor._spike_indices = np.array(data['indices'])
                monitor._total_spikes = len(data['times'])
                monitor.num_spikes = len(data['times'])
            else:
                monitor._spike_times = np.array([]) * ms
                monitor._spike_indices = np.array([])
                monitor._total_spikes = 0
                monitor.num_spikes = 0
        
        print("\nSimulation completed with memory management. Processing results")
        
        for name, mon in spike_monitors.items():
            print(f"{name} Total Number of Spikes: {mon.num_spikes}")
        
        if not spike_monitors:
            print("Warning: No spike monitors were created!")
            return results
        
        compute_firing_rates_all_neurons(spike_monitors, start_time=start_time, end_time=end_time, plot_order=plot_order)
        
        plot_raster(spike_monitors, sample_size=30, plot_order=plot_order, start_time=start_time, end_time=end_time)
        
        results = {
            'spike_monitors': spike_monitors,
            'firing_rates': firing_rates
        }
        
    except Exception as e:
        print(f"Simulation failed with error: {str(e)}")
        import traceback
        traceback.print_exc()
        raise
        
    finally:
        if spike_monitors:
            for mon in spike_monitors.values():
                if hasattr(mon, 'active'):
                    mon.active = False
        
        for obj in [spike_monitors, neuron_groups, synapse_connections, poisson_groups]:
            if obj is not None:
                del obj
        
        try:
            device.reinit()
            device.activate()
        except:
            pass
        
        gc.collect()
    
    return results