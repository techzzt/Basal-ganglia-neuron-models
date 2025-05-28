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
from module.models.Synapse import create_synapses, get_synapse_class
from module.utils.visualization import (
    plot_raster, plot_membrane_potential,
    plot_raster_all_neurons_stim_window
    )

from module.utils.sta import compute_firing_rates_all_neurons, adjust_connection_weights, estimate_required_weight_adjustment
from module.models.stimulus import create_poisson_inputs

# Matplotlib backend setup
# plt.ion() 

# Compiler optimization settings
os.environ['CC'] = 'gcc'
os.environ['CXX'] = 'g++'
prefs.codegen.target = 'cython'
prefs.codegen.cpp.extra_compile_args_gcc = ['-O3', '-ffast-math', '-march=native']
prefs.codegen.cpp.extra_compile_args_msvc = ['/O2']
prefs.devices.cpp_standalone.extra_make_args_unix = ['-j4'] 
prefs.devices.cpp_standalone.openmp_threads = 4 

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

def run_with_progress(net, duration, dt=1*ms, update_interval=500*ms):
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

def run_simulation_with_inh_ext_input(
    neuron_configs,
    connections,
    synapse_class,
    simulation_params,
    plot_order=None,
    start_time=0*ms,
    end_time=1000*ms,
    stim_pattern=None,
    ext_inputs=None
):
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
                sample_size = min(30, group.N)
                sample_indices = np.random.choice(group.N, size=sample_size, replace=False).tolist()
                sp_mon = SpikeMonitor(group, record=sample_indices)
                spike_monitors[name] = sp_mon
        
        net = Network(neuron_groups.values())
        net.add(synapse_connections)
        net.add(poisson_groups)
        net.add(spike_monitors.values())
        
        duration = simulation_params.get('duration', 2500) * ms
        dt = simulation_params.get('dt', 1) * ms
        
        print(f"\nStarting simulation for {duration/ms} ms")
        defaultclock.dt = dt
        
        chunk_size = 1000 * ms
        t = 0 * ms
        while t < duration:
            run_time = min(chunk_size, duration - t)
            print(f"Running simulation chunk: {t/ms} to {(t + run_time)/ms} ms")
            net.run(run_time)
            t += run_time
        for name, mon in spike_monitors.items():
            print(f"{name} 스파이크 개수: {mon.num_spikes}")
        print("\nSimulation completed. Processing results")
        
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
        
        gc.collect()
    
    return results

