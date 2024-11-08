from brian2 import *
import importlib


def create_synapses(neuron_groups, synapse_params, synapse_class):

    try:
        module = importlib.import_module(f'Neuronmodels.{synapse_class}')
        synapse_class = getattr(module, synapse_class)
    except (ImportError, AttributeError) as e:
        raise ImportError(f"Synapse Class {synapse_class} Error: {str(e)}")
    
    try:
        synapse_instance = synapse_class(neuron_groups, synapse_params)
        synapse_connections = synapse_instance.create_synapse()
        return synapse_connections
    except Exception as e:
        raise Exception(f"Synapse Error: {str(e)}")

class SynapseBase:
    def __init__(self, neurons, params):
        self.neurons = neurons
        for key, value in neurons.items():
            setattr(self, key, value)
        self.params = params
        
    def _get_param(self, param_name):
        if param_name not in self.params:
            raise KeyError(f"Parameter {param_name} not found")
            
        value = self.params[param_name]['value']
        unit = self.params[param_name]['unit']
        
        # 단위 변환
        if unit == 'ms':
            return value * ms
        elif unit == 'mV':
            return value * mV
        elif unit == 'nS':
            return value * nS
        elif unit == 'pA':
            return value * pA
        else:
            return value * eval(unit)  
            
    def create_synapse(self):
        raise NotImplementedError