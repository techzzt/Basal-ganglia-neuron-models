from brian2 import *
import importlib
from module.models import AdEx
from module.models.Synapse import generate_neuron_specific_synapse_inputs
import numpy as np
import bisect
from brian2 import mV, ms, nS, pA

from brian2.utils.arrays import calc_repeats
from brian2.utils.logger import get_logger

class NeuronModel:
    def __init__(self, N, params):
        self.N = N
        self.params = params
        self.neurons = None

    def create_neurons(self):
        raise NotImplementedError("Subclasses should implement this method.")

class STN(NeuronModel):
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
            if conn_data['post'] == "STN":  
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
            synapse_eqs = generate_neuron_specific_synapse_inputs("STN", self.connections)
            full_eqs = base_eqs + '\n' + synapse_eqs
        else:
            full_eqs = base_eqs + '\nIsyn = 0*amp : amp\n'
        
        reset = '''
        v_reset_val = vr 
        v_reset_applied = v_reset_val + clip(z - 15*pA, 20*pA, 1000*pA) / nS
        v = v_reset_applied * int(z < 0*pA) + v * int(z >= 0*pA)
        z += d
        '''
        
        self.neurons = NeuronGroup(
            self.N, full_eqs, threshold='v > th', reset=reset, method='euler'
        )

        self.neurons.g_L = self.params['g_L']['value'] * eval(self.params['g_L']['unit'])
        self.neurons.E_L = self.params['E_L']['value'] * eval(self.params['E_L']['unit'])
        self.neurons.Delta_T = self.params['Delta_T']['value'] * eval(self.params['Delta_T']['unit'])
        self.neurons.vt = self.params['vt']['value'] * eval(self.params['vt']['unit'])
        self.neurons.vr = self.params['vr']['value'] * eval(self.params['vr']['unit'])
        np.random.seed(2025) 
        base_v = self.params['v']['value'] * eval(self.params['v']['unit'])
        v_noise = np.random.normal(0, 2, self.N) * mV 
        self.neurons.v = base_v + v_noise
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
            
        if 'AMPA' in self.receptor_params:
            ampa_param_list = self.receptor_params['AMPA']
            dominant_params = max(ampa_param_list, key=lambda p: p['g0']['value'])
            
            if hasattr(self.neurons, 'tau_AMPA'):
                self.neurons.tau_AMPA = dominant_params['tau_syn']['value'] * eval(dominant_params['tau_syn']['unit'])
            if hasattr(self.neurons, 'E_AMPA'):
                self.neurons.E_AMPA = dominant_params['E_rev']['value'] * eval(dominant_params['E_rev']['unit'])
            if hasattr(self.neurons, 'ampa_beta'):
                self.neurons.ampa_beta = dominant_params.get('beta', {'value': 1.0})['value']
        else:
            if hasattr(self.neurons, 'tau_AMPA'):
                self.neurons.tau_AMPA = 12 * ms
            if hasattr(self.neurons, 'E_AMPA'):
                self.neurons.E_AMPA = 0 * mV
            if hasattr(self.neurons, 'ampa_beta'):
                self.neurons.ampa_beta = 1.0
                
        if 'GABA' in self.receptor_params:
            gaba_param_list = self.receptor_params['GABA']
            dominant_params = max(gaba_param_list, key=lambda p: p['g0']['value'])
            
            if hasattr(self.neurons, 'tau_GABA'):
                self.neurons.tau_GABA = dominant_params['tau_syn']['value'] * eval(dominant_params['tau_syn']['unit'])
            if hasattr(self.neurons, 'E_GABA'):
                self.neurons.E_GABA = dominant_params['E_rev']['value'] * eval(dominant_params['E_rev']['unit'])
            if hasattr(self.neurons, 'gaba_beta'):
                self.neurons.gaba_beta = dominant_params.get('beta', {'value': 1.0})['value']
        else:
            if hasattr(self.neurons, 'tau_GABA'):
                self.neurons.tau_GABA = 6 * ms
            if hasattr(self.neurons, 'E_GABA'):
                self.neurons.E_GABA = -74 * mV
            if hasattr(self.neurons, 'gaba_beta'):
                self.neurons.gaba_beta = 1.0

        return self.neurons