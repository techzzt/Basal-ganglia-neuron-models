import importlib
from brian2 import *
from module.models import QIF
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

class MSND1(NeuronModel):
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
            if conn_data['post'] == "MSND1":  
                conn_receptor_params = conn_data.get('receptor_params', {})
                
                for receptor_type, params in conn_receptor_params.items():
                    if receptor_type not in receptor_params:
                        receptor_params[receptor_type] = [params]  
                    else:
                        receptor_params[receptor_type].append(params) 
        return receptor_params

    def create_neurons(self):
        base_eqs = QIF.eqs 
        
        if self.connections:
            synapse_eqs = generate_neuron_specific_synapse_inputs("MSND1", self.connections)
            full_eqs = base_eqs + '\n' + synapse_eqs
        else:
            full_eqs = base_eqs + '\nIsyn = 0*amp : amp\n'
        
        self.neurons = NeuronGroup(self.N, full_eqs, threshold='v > th', reset='v = c; u += d', method='euler')
        
        self.neurons.a = self.params['a']['value'] * eval(self.params['a']['unit'])
        self.neurons.b = self.params['b']['value'] 
        self.neurons.C = self.params['C']['value'] * eval(self.params['C']['unit'])
        self.neurons.c = self.params['c']['value'] * eval(self.params['c']['unit'])
        self.neurons.d = self.params['d']['value'] * eval(self.params['d']['unit'])
        self.neurons.k = self.params['k']['value'] 

        self.neurons.vr = self.params['vr']['value'] * eval(self.params['vr']['unit'])
        self.neurons.vt = self.params['vt']['value'] * eval(self.params['vt']['unit'])
        self.neurons.v = self.params['v']['value'] * eval(self.params['v']['unit'])
        self.neurons.th = self.params['th']['value'] * eval(self.params['th']['unit'])
        
        self.neurons.u = self.params['u']['value'] * eval(self.params['u']['unit'])
        
        self.neurons.I_ext = 0 * pA

        if hasattr(self.neurons, 'g_a'):
            self.neurons.g_a = 0 * nS
        if hasattr(self.neurons, 'g_g'):
            self.neurons.g_g = 0 * nS
        if hasattr(self.neurons, 'g_n'):
            self.neurons.g_n = 0 * nS
            
        if 'AMPA' in self.receptor_params:
            params = self.receptor_params['AMPA'][0] 
            if hasattr(self.neurons, 'tau_AMPA'):
                self.neurons.tau_AMPA = params['tau_syn']['value'] * eval(params['tau_syn']['unit'])
            if hasattr(self.neurons, 'E_AMPA'):
                self.neurons.E_AMPA = params['E_rev']['value'] * eval(params['E_rev']['unit'])
            if hasattr(self.neurons, 'ampa_beta'):
                self.neurons.ampa_beta = params.get('beta', {'value': 1.0})['value']
                
        if 'GABA' in self.receptor_params:
            params = self.receptor_params['GABA'][0]  
            
            if hasattr(self.neurons, 'tau_GABA'):
                self.neurons.tau_GABA = params['tau_syn']['value'] * eval(params['tau_syn']['unit'])
            if hasattr(self.neurons, 'E_GABA'):
                self.neurons.E_GABA = params['E_rev']['value'] * eval(params['E_rev']['unit'])
            if hasattr(self.neurons, 'gaba_beta'):
                self.neurons.gaba_beta = params.get('beta', {'value': 1.0})['value']

        if 'NMDA' in self.receptor_params:
            params = self.receptor_params['NMDA'][0] 
            
            if hasattr(self.neurons, 'tau_NMDA'):
                self.neurons.tau_NMDA = params['tau_syn']['value'] * eval(params['tau_syn']['unit'])
            if hasattr(self.neurons, 'E_NMDA'):
                self.neurons.E_NMDA = params['E_rev']['value'] * eval(params['E_rev']['unit'])
            if hasattr(self.neurons, 'nmda_beta'):
                self.neurons.nmda_beta = params.get('beta', {'value': 1.0})['value']
            if hasattr(self.neurons, 'Mg2'):
                self.neurons.Mg2 = 1.0  

        return self.neurons