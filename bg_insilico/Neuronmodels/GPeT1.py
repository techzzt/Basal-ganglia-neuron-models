from brian2 import *
import importlib
from module.models import AdEx
from brian2 import mV, ms, nS
import numpy as np 

class NeuronModel:
    def __init__(self, N, params):
        super().__init__(N, params)
        self.neurons = None

    def create_neurons(self):
        raise NotImplementedError("Subclasses should implement this method.")

class GPeT1(NeuronModel):
    def __init__(self, N, params, connections=None):  
        self.N = N
        self.params = params
        self.receptor_params = self.get_receptor_params(connections) if connections else {} 
        self.neurons = None

    def get_receptor_params(self, connections):
        receptor_params = {}
        for conn_name, conn_data in connections.items():
            if conn_data['post'] == "GPeT1":  
                conn_receptor_params = conn_data.get('receptor_params', {})
                
                for receptor_type, params in conn_receptor_params.items():
                    if receptor_type not in receptor_params:
                        receptor_params[receptor_type] = [params]  
                    else:
                        receptor_params[receptor_type].append(params) 
        return receptor_params

    def create_neurons(self):
        eqs = AdEx.eqs 

        self.neurons = NeuronGroup(
            self.N, eqs, threshold='v > th', reset='v = vr; z += d', method='euler'
        )
        self.neurons.g_L = self.params['g_L']['value'] * eval(self.params['g_L']['unit'])
        self.neurons.E_L = self.params['E_L']['value'] * eval(self.params['E_L']['unit'])
        self.neurons.Delta_T = self.params['Delta_T']['value'] * eval(self.params['Delta_T']['unit'])
        self.neurons.vt = self.params['vt']['value'] * eval(self.params['vt']['unit'])
        self.neurons.vr = self.params['vr']['value'] * eval(self.params['vr']['unit'])
        vr_val = self.params['vr']['value'] 
        vt_val = self.params['vt']['value']
        v_random = np.random.uniform(vr_val, vt_val, self.N)
        self.neurons.v = v_random * mV
        self.neurons.tau_w = self.params['tau_w']['value'] * eval(self.params['tau_w']['unit'])
        self.neurons.th = self.params['th']['value'] * eval(self.params['th']['unit'])
        self.neurons.a = self.params['a']['value'] * eval(self.params['a']['unit'])
        self.neurons.d = self.params['d']['value'] * eval(self.params['d']['unit'])
        self.neurons.C = self.params['C']['value'] * eval(self.params['C']['unit'])
        self.neurons.I_ext = self.params['I_ext']['value'] * eval(self.params['I_ext']['unit'])
        self.neurons.z = self.params['z']['value'] * eval(self.params['z']['unit'])

        return self.neurons
