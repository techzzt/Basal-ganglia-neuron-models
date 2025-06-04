from brian2 import *
import importlib
from module.models import AdEx
from module.models.Synapse import generate_neuron_specific_synapse_inputs
from brian2 import mV, ms, nS
import numpy as np 

class NeuronModel:
    def __init__(self, N, params):
        self.N = N
        self.params = params
        self.neurons = None

    def create_neurons(self):
        raise NotImplementedError("Subclasses should implement this method.")

class GPeTA(NeuronModel):
    def __init__(self, N, params, connections=None):  
        super().__init__(N, params)
        self.N = N
        self.params = params
        self.connections = connections
        self.receptor_params = self.get_receptor_params(connections) if connections else {}
        self.neurons = None

    def get_receptor_params(self, connections):
        receptor_params = {}
        for conn_name, conn_data in connections.items():
            if conn_data['post'] == "GPeTA":  
                conn_receptor_params = conn_data.get('receptor_params', {})
                
                for receptor_type, params in conn_receptor_params.items():
                    if receptor_type not in receptor_params:
                        receptor_params[receptor_type] = [params]  
                    else:
                        receptor_params[receptor_type].append(params) 
        return receptor_params

    def create_neurons(self):
        base_eqs = AdEx.eqs 

        if self.connections:
            synapse_eqs = generate_neuron_specific_synapse_inputs("GPeTA", self.connections)
            full_eqs = base_eqs + '\n' + synapse_eqs
        else:
            full_eqs = base_eqs + '\nIsyn = 0*amp : amp\n'

        self.neurons = NeuronGroup(
            self.N, full_eqs, threshold='v > th', reset='v = vr; z += d', method='euler'
        )
        self.neurons.g_L = self.params['g_L']['value'] * eval(self.params['g_L']['unit'])
        self.neurons.E_L = self.params['E_L']['value'] * eval(self.params['E_L']['unit'])
        self.neurons.Delta_T = self.params['Delta_T']['value'] * eval(self.params['Delta_T']['unit'])
        self.neurons.vt = self.params['vt']['value'] * eval(self.params['vt']['unit'])
        self.neurons.vr = self.params['vr']['value'] * eval(self.params['vr']['unit'])
        self.neurons.v = self.params['v']['value'] * eval(self.params['v']['unit'])
        self.neurons.tau_w = self.params['tau_w']['value'] * eval(self.params['tau_w']['unit'])
        self.neurons.th = self.params['th']['value'] * eval(self.params['th']['unit'])
        self.neurons.a = self.params['a']['value'] * eval(self.params['a']['unit'])
        self.neurons.d = self.params['d']['value'] * eval(self.params['d']['unit'])
        self.neurons.C = self.params['C']['value'] * eval(self.params['C']['unit'])
        self.neurons.I_ext = self.params['I_ext']['value'] * eval(self.params['I_ext']['unit'])
        self.neurons.z = self.params['z']['value'] * eval(self.params['z']['unit'])

        if hasattr(self.neurons, 'g_a'):
            self.neurons.g_a = 0 * nS
        if hasattr(self.neurons, 'g_g'):
            self.neurons.g_g = 0 * nS
        if hasattr(self.neurons, 'g_n'):
            self.neurons.g_n = 0 * nS
            
        if hasattr(self.neurons, 'tau_AMPA'):
            self.neurons.tau_AMPA = 12 * ms
        if hasattr(self.neurons, 'E_AMPA'):
            self.neurons.E_AMPA = 0 * mV
        if hasattr(self.neurons, 'ampa_beta'):
            self.neurons.ampa_beta = 1.0
        if hasattr(self.neurons, 'tau_GABA'):
            self.neurons.tau_GABA = 6 * ms
        if hasattr(self.neurons, 'E_GABA'):
            self.neurons.E_GABA = -74 * mV
        if hasattr(self.neurons, 'gaba_beta'):
            self.neurons.gaba_beta = 1.0
            
        return self.neurons