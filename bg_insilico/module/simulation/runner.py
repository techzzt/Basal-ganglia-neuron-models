import numpy as np
import os
from copy import deepcopy
import time
from tqdm import tqdm
import math

from brian2 import *
from module.models.neuron_models import create_neurons
from module.models.Synapse import create_synapses, get_synapse_class
from module.utils.visualization import (
    plot_raster, plot_membrane_potential,
    plot_raster_all_neurons_stim_window
    )

from module.utils.sta import compute_firing_rates_all_neurons, adjust_connection_weights, estimate_required_weight_adjustment
from module.models.stimulus import create_poisson_inputs

os.environ['CC'] = 'gcc'
os.environ['CXX'] = 'g++'

prefs.codegen.target = 'cython'
prefs.codegen.cpp.extra_compile_args_gcc = ['-O3', '-ffast-math', '-march=native']
prefs.codegen.cpp.extra_compile_args_msvc = ['/O2']
prefs.devices.cpp_standalone.openmp_threads = 4  

class SimulationMonitor:
    
    def __init__(self, total_time, dt=1*ms, update_interval=100*ms):
        self.total_time = float(total_time/second)
        self.start_time = time.time()
        self.last_t = 0
        self.dt = float(dt/second)
        self.update_interval = float(update_interval/second)
        self.total_steps = int(self.total_time / self.dt)
        self.pbar = tqdm(total=self.total_time, unit='s', desc='Simulation Progress')
    
    def update(self, t):
        current_time = float(t/second)
        progress = current_time - self.last_t
        
        if progress > 0:
            self.pbar.update(progress)
            self.last_t = current_time
            
            elapsed = time.time() - self.start_time
            total_progress_fraction = current_time / self.total_time
            
            if total_progress_fraction > 0:
                estimated_total = elapsed / total_progress_fraction
                remaining = max(0, estimated_total - elapsed)
                self.pbar.set_postfix(remaining=f"{remaining:.1f}s")
    
    def close(self):
        self.pbar.close()

def run_with_progress(net, duration, dt=0.1*ms, update_interval=100*ms):
    monitor = SimulationMonitor(duration, dt=dt, update_interval=update_interval)
    
    @network_operation(dt=update_interval)
    def update_progress(t):
        monitor.update(t)
    
    net.add(update_progress)
    
    try:
        net.run(duration, report=None)
    finally:
        monitor.close()

def run_simulation_with_inh_ext_input(neuron_configs, connections, simulation_params, stim_pattern=None, ext_inputs=None):
    neuron_groups = create_neurons(neuron_configs, simulation_params, connections)
    synapse_class = get_synapse_class('Synapse')
    synapse_connections = create_synapses(neuron_groups, connections, synapse_class)
    poisson_groups = create_poisson_inputs(neuron_groups, ext_inputs) if ext_inputs else []
    
    monitors = []
    for name, group in neuron_groups.items():
        if isinstance(group, (PoissonGroup, SpikeGeneratorGroup)):
            spike_monitor = SpikeMonitor(group)
            monitors.append(spike_monitor)
        else:
            try:
                n_record = max(1, int(len(group) * 0.1))
                record_indices = np.random.choice(len(group), n_record, replace=False)
                
                recordable_variables = []
                if 'v' in group.variables:
                    recordable_variables.append('v')
                if 'I' in group.variables:
                    recordable_variables.append('I')
                
                if recordable_variables:
                    state_monitor = StateMonitor(group, recordable_variables, record=record_indices)
                    monitors.append(state_monitor)
                    print(f"Created state monitor for {name} recording {recordable_variables}")
                
                spike_monitor = SpikeMonitor(group)
                monitors.append(spike_monitor)
            except Exception as e:
                print(f"Warning: Could not create full monitors for {name}: {str(e)}")
                spike_monitor = SpikeMonitor(group)
                monitors.append(spike_monitor)
    
    net = Network(neuron_groups.values())
    net.add(*synapse_connections)
    net.add(*poisson_groups)
    net.add(*monitors)
    
    run_time = simulation_params.get('run_time', 1) * second
    dt = simulation_params.get('dt', 1) * ms
    
    print(f"Starting simulation: {run_time/ms} ms, dt={dt/ms} ms")
    defaultclock.dt = dt
    
    run_with_progress(net, run_time, dt=dt, update_interval=100*ms)
    
    results = {
        'neuron_groups': neuron_groups,
        'monitors': {}
    }
    
    monitor_index = 0
    for name, group in neuron_groups.items():
        results['monitors'][name] = {}
        
        if isinstance(group, (PoissonGroup, SpikeGeneratorGroup)):
            results['monitors'][name]['spike'] = monitors[monitor_index]
            monitor_index += 1
        else:
            if monitor_index < len(monitors) and isinstance(monitors[monitor_index], StateMonitor):
                results['monitors'][name]['state'] = monitors[monitor_index]
                monitor_index += 1
            if monitor_index < len(monitors) and isinstance(monitors[monitor_index], SpikeMonitor):
                results['monitors'][name]['spike'] = monitors[monitor_index]
                monitor_index += 1
    
    return results

