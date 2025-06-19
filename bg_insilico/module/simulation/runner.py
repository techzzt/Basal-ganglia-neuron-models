import numpy as np
import os
from copy import deepcopy
import time
from tqdm import tqdm
import gc
import random as python_random

from brian2 import *
from brian2 import mV, ms, nS, Hz
from module.models.neuron_models import create_neurons
from module.models.Synapse import create_synapses
from module.utils.visualization import (
    plot_raster, plot_membrane_potential,
    plot_raster_all_neurons_stim_window
    )

from module.utils.sta import compute_firing_rates_all_neurons
from module.models.stimulus import create_poisson_inputs
from brian2.devices.device import reset_device
from module.utils.param_loader import load_params

# Matplotlib backend setup
# plt.ion() 

try:
    reset_device()
except:
    pass

import platform
if platform.system() == 'Darwin':
    os.environ['CC'] = '/usr/bin/clang'
    os.environ['CXX'] = '/usr/bin/clang++'

prefs.codegen.target = 'cython'
prefs.core.default_float_dtype = np.float32
prefs.codegen.runtime.cython.multiprocess_safe = False
prefs.codegen.runtime.numpy.discard_units = True

class SimulationMonitor:
    def __init__(self, total_time, dt=1*ms, update_interval=100*ms):  
        self.total_time = float(total_time/ms)  
        self.start_time = time.time()
        self.last_t = 0
        self.dt = float(dt/ms)
        self.update_interval = float(update_interval/ms)
        self.total_steps = int(self.total_time / self.dt)
        self.pbar = tqdm(total=self.total_time, unit='ms', desc='Simulation Progress')
    
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
            pass
    
    def close(self):
        self.pbar.close()


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


def run_simulation_with_inh_ext_input(
    neuron_configs,
    connections,
    synapse_class,
    simulation_params,
    plot_order=None,
    start_time=0*ms,
    end_time=1000*ms,
    ext_inputs=None,
    amplitude_oscillations=None
):
    
    spike_monitors = {}
    neuron_groups = None
    synapse_connections = None
    poisson_groups = None
    firing_rates = {}
    results = {'spike_monitors': {}, 'firing_rates': {}}
    net = None
    duration = None
    
    try:
        device.reinit()
        device.activate()
        start_scope()
        for name in list(globals().keys()):
            if name.startswith('_'):
                del globals()[name]
        
        net = Network()
        neuron_groups = create_neurons(neuron_configs, simulation_params, connections)

        synapse_connections = create_synapses(neuron_groups, connections, synapse_class)
        
        scaled_neuron_counts = {name: group.N for name, group in neuron_groups.items()}
        poisson_groups, _ = create_poisson_inputs(
            neuron_groups, 
            ext_inputs, 
            scaled_neuron_counts,
            amplitude_oscillations
        ) if ext_inputs else ({}, [])
        
        for name, group in neuron_groups.items():
            if not name.startswith(('Cortex_', 'Ext_')):  
                sp_mon = SpikeMonitor(group)
                spike_monitors[name] = sp_mon
        
        net.add(*neuron_groups.values())
        net.add(*synapse_connections)
        net.add(*poisson_groups.values())
        net.add(*spike_monitors.values())
        
        duration = simulation_params.get('duration', 1000) * ms
        dt = simulation_params.get('dt', 0.1) * ms 
        defaultclock.dt = dt
        
        report_interval = duration * 0.5
        net.run(duration, report='text', report_period=report_interval)
        
        if not spike_monitors:
            return results
        
        firing_rates = compute_firing_rates_all_neurons(spike_monitors, start_time=start_time, end_time=end_time, plot_order=plot_order)
        display_names = simulation_params.get('display_names', None)

        plot_raster(spike_monitors, sample_size=30, plot_order=plot_order, start_time=start_time, end_time=end_time, display_names=display_names)
        
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