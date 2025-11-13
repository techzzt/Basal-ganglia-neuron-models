# -*- coding: utf-8 -*-
# Copyright (c) 2025. All rights reserved.
# Author: keun (Jieun Kim)

from brian2 import *
import importlib
import json

# Load parameters from JSON
def load_params_from_file(params_file):
    with open(params_file, 'r') as f:
        data = json.load(f)
        return data['params']

# Get neuron count by name
def get_neuron_count(neuron_configs, target_name):
    for config in neuron_configs:
        if config["name"] == target_name:
            return config["N"]
    return None  

# Create neuron groups
def create_neurons(neuron_configs, simulation_params, connections=None):
    neuron_groups = {}

    for config in neuron_configs:
        if config.get('neuron_type') == 'poisson':
            continue

        if 'model_class' not in config or 'params_file' not in config:
            continue

        name = config['name']
        N = config['N']
        
        params = load_params_from_file(config['params_file'])
        module_name = f"Neuronmodels.{config['model_class']}"
        model_module = importlib.import_module(module_name)
        model_class = getattr(model_module, config['model_class'])
        model_instance = model_class(N, params, connections)
        neuron_group = model_instance.create_neurons()
        neuron_groups[name] = neuron_group

    return neuron_groups