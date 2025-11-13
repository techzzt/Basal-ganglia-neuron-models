# -*- coding: utf-8 -*-
# Copyright (c) 2025. All rights reserved.
# Author: keun (Jieun Kim)
# Description: Parameter validation before full simulation

from brian2 import *
from brian2 import mV, ms, pA, nS
import numpy as np
import json
import os
import copy
import importlib
import sys
from module.utils.param_loader import load_params
from module.simulation.runner import run_simulation_with_inh_ext_input
from module.utils.firing_rate_analysis import calculate_and_print_firing_rates

# Target firing rate ranges (Hz)
TARGET_RATES = {
    'FSN': (10, 20),
    'STN': (10, 20),
    'STN_PVminus': (10, 20),
    'STN_PVplus': (10, 20),
    'MSND1': (0.02, 2),
    'MSND2': (0.02, 2),
    'SNr': (20, 35),
    'GPeT1': (18, 18),   
    'GPeTA': (8, 11)    
}

def estimate_firing_rate_single_neuron(neuron_type, params_file, duration_ms=1000, dt_ms=0.1, I_ext_override=None):

    start_scope()
    
    try:
        params_file = os.path.abspath(params_file)
        with open(params_file, 'r') as f:
            param_data = json.load(f)
        
        params = param_data['params'].copy()  
        module_name = f"Neuronmodels.{neuron_type}"
        if module_name in sys.modules:
            importlib.reload(sys.modules[module_name])
        model_module = importlib.import_module(module_name)
        model_class = getattr(model_module, neuron_type)
        model_instance = model_class(1, params, connections=None)  # Single neuron
        neuron_group = model_instance.create_neurons()
        
        # Set up simulation
        defaultclock.dt = dt_ms * ms
        
        # Monitor spikes
        spike_monitor = SpikeMonitor(neuron_group)
        
        # Create network and run
        net = Network()
        net.add(neuron_group, spike_monitor)
        net.run(duration_ms * ms)
        
        # Calculate firing rate
        if len(spike_monitor.t) > 0:
            duration_sec = duration_ms / 1000.0
            firing_rate = len(spike_monitor.t) / duration_sec
        else:
            firing_rate = 0.0
        
        return firing_rate
        
    except Exception as e:
        print(f"Error estimating firing rate for {neuron_type}: {e}")
        return None

def estimate_firing_rate_with_network(config_file, target_neuron_name=None, duration_ms=2000, dt_ms=0.1, skip_warmup_ms=500, scale_population=0.1):

    start_scope()
    
    try:
        # Load config
        params = load_params(config_file)
        neuron_configs = params['neurons']
        connections = params['connections']
        synapse_class = params['synapse_class']
        simulation_params = params.get('simulation', {})
        
        # Scale down population sizes for quick test
        scaled_neuron_configs = copy.deepcopy(neuron_configs)
        for config in scaled_neuron_configs:
            if 'N' in config:
                original_N = config['N']
                scaled_N = max(1, int(original_N * scale_population))
                config['N'] = scaled_N
        
        # Use shorter duration for quick test
        test_simulation_params = simulation_params.copy()
        test_simulation_params['duration'] = duration_ms
        test_simulation_params['dt'] = dt_ms
        
        # Setup external inputs
        ext_inputs = {}
        for neuron_config in neuron_configs:
            if neuron_config.get('neuron_type') == 'poisson':
                name = neuron_config['name']
                if 'target_rates' in neuron_config:
                    target, rate_info = list(neuron_config['target_rates'].items())[0]
                    rate_expr = rate_info['equation']
                    ext_inputs[target] = rate_expr
        
        amplitude_oscillations = {
            'MSND1': 0.11,
            'MSND2': 0.11,
            'FSN': 0.11,
            'STN': 0.11,
            'STN_PVminus': 0.11,
            'STN_PVplus': 0.11
        }
        
        print(f"Running network simulation (duration: {duration_ms}ms, warmup: {skip_warmup_ms}ms, population: {scale_population*100:.0f}%)...")
        results = run_simulation_with_inh_ext_input(
            neuron_configs=scaled_neuron_configs,
            connections=connections,
            synapse_class=synapse_class,
            simulation_params=test_simulation_params,
            plot_order=None,
            start_time=skip_warmup_ms * ms,
            end_time=duration_ms * ms,
            ext_inputs=ext_inputs,
            amplitude_oscillations=amplitude_oscillations
        )
        
        # Calculate firing rates
        firing_rates = calculate_and_print_firing_rates(
            results['spike_monitors'],
            start_time=skip_warmup_ms * ms,
            end_time=duration_ms * ms,
            display_names=params.get('display_names', None),
            skip_warmup_ms=skip_warmup_ms
        )
        
        if target_neuron_name:
            return {target_neuron_name: firing_rates.get(target_neuron_name, 0.0)}
        
        return firing_rates
        
    except Exception:
        return {}

def validate_parameters(config_file, skip_warmup_ms=2000, use_network=True):

    if use_network:
        print("=" * 60)
        print("Parameter Validation (Network Simulation)")
        print("=" * 60)

    else:
        print("=" * 60)
        print("Parameter Validation (Single Neuron Estimation)")
        print("=" * 60)
    
    try:
        params = load_params(config_file)
        neuron_configs = params['neurons']
    except Exception as e:
        print(f"Error loading config file: {e}")
        return {'all_valid': False, 'error': str(e)}
    
    config_file_abs = os.path.abspath(config_file)
    config_dir = os.path.dirname(config_file_abs)

    if os.path.basename(config_dir) == 'config':
        base_dir = os.path.dirname(config_dir)
    else:
        base_dir = config_dir
    if not base_dir:
        base_dir = os.getcwd()
    
    results = {}
    all_valid = True
    
    if use_network:
        # Use network simulation with scaled-down population
        firing_rates = estimate_firing_rate_with_network(
            config_file,
            duration_ms=2000,
            dt_ms=0.1,
            skip_warmup_ms=500,
            scale_population=0.1
        )
        
        for neuron_config in neuron_configs:
            if neuron_config.get('neuron_type') == 'poisson':
                continue
            
            name = neuron_config['name']
            firing_rate = firing_rates.get(name, 0.0)
            
            target_range = TARGET_RATES.get(name, (0, 1000))
            min_rate, max_rate = target_range
            
            is_valid = min_rate <= firing_rate <= max_rate
            status = "Y" if is_valid else "N"
            
            if not is_valid:
                all_valid = False
                if firing_rate < min_rate:
                    suggestion = f"Increase v (or I_ext) to raise firing rate"
                else:
                    suggestion = f"Decrease v (or I_ext) to lower firing rate"
            else:
                suggestion = "Within target range"
            
            print(f"\n{status} {name}: {firing_rate:.3f} Hz (target: {min_rate:.2f}-{max_rate:.2f} Hz)")
            print(f"   Suggestion: {suggestion}")
            
            results[name] = {
                'valid': is_valid,
                'firing_rate': firing_rate,
                'target_range': target_range,
                'suggestion': suggestion
            }
    else:
        # Use single neuron simulation
        for neuron_config in neuron_configs:
            if neuron_config.get('neuron_type') == 'poisson':
                continue
            
            name = neuron_config['name']
            model_class = neuron_config.get('model_class', name)
            params_file = neuron_config.get('params_file', '')
            
            # Resolve relative paths
            if params_file and not os.path.isabs(params_file):
                if params_file.startswith('./'):
                    params_file = params_file[2:]
                params_file = os.path.join(base_dir, params_file)
                params_file = os.path.normpath(params_file)
            
            if not params_file or not os.path.exists(params_file):
                print(f"{name}: Parameter file not found: {params_file}")
                results[name] = {'valid': False, 'reason': 'Parameter file not found'}
                all_valid = False
                continue
            
            # Estimate firing rate
            print(f"\nEstimating firing rate for {name}...")
            firing_rate = estimate_firing_rate_single_neuron(
                model_class, 
                params_file,
                duration_ms=1000,
                dt_ms=0.1
            )
            
            if firing_rate is None:
                print(f"{name}: Failed to estimate firing rate")
                results[name] = {'valid': False, 'reason': 'Estimation failed'}
                all_valid = False
                continue
            
            target_range = TARGET_RATES.get(name, (0, 1000))
            min_rate, max_rate = target_range
            
            is_valid = min_rate <= firing_rate <= max_rate
            status = "Y" if is_valid else "N"
            
            if not is_valid:
                all_valid = False
                if firing_rate < min_rate:
                    suggestion = f"Increase v (or I_ext) to raise firing rate"
                else:
                    suggestion = f"Decrease v (or I_ext) to lower firing rate"
            else:
                suggestion = "Within target range"
            
            print(f"{status} {name}: {firing_rate:.3f} Hz (target: {min_rate:.2f}-{max_rate:.2f} Hz)")
            print(f"   Suggestion: {suggestion}")
            
            results[name] = {
                'valid': is_valid,
                'firing_rate': firing_rate,
                'target_range': target_range,
                'suggestion': suggestion
            }
        
    print("\n" + "=" * 60)
    if all_valid:
        print("All parameters are within target ranges")
    else:
        print("Some parameters are outside target ranges")
        print("Please adjust parameters before running full simulation.")
    print("=" * 60)
    
    return {
        'all_valid': all_valid,
        'results': results
    }

if __name__ == "__main__":
    if len(sys.argv) > 1:
        config_file = sys.argv[1]
        validate_parameters(config_file)
    else:
        print("Usage: python parameter_validator.py <config_file>")
