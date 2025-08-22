# -*- coding: utf-8 -*-
# Copyright (c) 2025. All rights reserved.
# Author: keun (Jieun Kim)
# Description: Neuron group management

from brian2 import *
import importlib
import json
import numpy as np 
from brian2 import mV, ms, nS, Hz
import gc  
from module.models.stimulus import create_poisson_inputs

# Load parameters from JSON
def load_params_from_file(params_file):
    try:
        with open(params_file, 'r') as f:
            data = json.load(f)
            return data['params']
    except Exception as e:
        print(f"Load Error: {str(e)}")
        raise

# Get neuron count by name
def get_neuron_count(neuron_configs, target_name):
    for config in neuron_configs:
        if config["name"] == target_name:
            return config["N"]
    return None  

# Create neuron groups
def create_neurons(neuron_configs, simulation_params, connections=None):
    duration = simulation_params.get('duration', 1.0) 
    dt = float(simulation_params.get('dt', 1))  
    stimulus_enabled = simulation_params.get('stimulus', {}).get('enabled', False)

    try:
        neuron_groups = {}

        for config in neuron_configs:
            name = config['name']
            N = config['N']

            if config.get('neuron_type', None) == 'poisson':
                continue

            if 'model_class' in config and 'params_file' in config:
                try:
                    params = load_params_from_file(config['params_file'])
                    module_name = f"Neuronmodels.{config['model_class']}"
                    model_module = importlib.import_module(module_name)
                    model_class = getattr(model_module, config['model_class'])
                    model_instance = model_class(N, params, connections)
                    neuron_group = model_instance.create_neurons()
                    neuron_groups[name] = neuron_group
                except Exception as e:
                    print(f"Error creating {name}: {str(e)}")
                    import traceback
                    traceback.print_exc()
                    raise

        return neuron_groups

    except Exception as e:
        print(f"Error creating neuron groups: {str(e)}")
        print(f"Failed configuration: {config}")
        raise