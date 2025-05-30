import importlib
from brian2 import *
from brian2 import mV, ms, nS
import numpy as np 

from module.models import QIF_FSN

class NeuronModel:
    def __init__(self, N, params):
        super().__init__()  
        self.N = N
        self.params = params
        self.neurons = None

    def create_neurons(self):
        raise NotImplementedError("Subclasses should implement this method.")

class FSN(NeuronModel):
    def __init__(self, N, params, connections=None):  
        super().__init__(N, params) 
        self.N = N
        self.params = params
        self.receptor_params = self.get_receptor_params(connections) if connections else {}
        self.neurons = None
    
    def get_receptor_params(self, connections):
        receptor_params = {}
        for conn_name, conn_data in connections.items():
            if conn_data['post'] == "FSN":  
                conn_receptor_params = conn_data.get('receptor_params', {})
                
                for receptor_type, params in conn_receptor_params.items():
                    if receptor_type not in receptor_params:
                        receptor_params[receptor_type] = [params]  
                    else:
                        receptor_params[receptor_type].append(params) 
        return receptor_params
    
    def create_neurons(self):
        eqs = QIF_FSN.eqs 
        reset_eqs = '''
        v = c;
        u += d
        '''
        self.neurons = NeuronGroup(
            self.N, eqs, threshold='v > th', reset = reset_eqs, method='euler'
        )
        self.neurons.vr = self.params['vr']['value'] * eval(self.params['vr']['unit'])
        self.neurons.vt = self.params['vt']['value'] * eval(self.params['vt']['unit'])
        
        # Use fixed initial membrane potential from JSON parameters
        self.neurons.v = self.params['v']['value'] * eval(self.params['v']['unit'])
        
        self.neurons.th = self.params['th']['value'] * eval(self.params['th']['unit'])
        self.neurons.k = self.params['k']['value'] 
        self.neurons.a = self.params['a']['value'] * eval(self.params['a']['unit'])
        self.neurons.b = self.params['b']['value'] 
        self.neurons.C = self.params['C']['value'] * eval(self.params['C']['unit'])
        self.neurons.c = self.params['c']['value'] * eval(self.params['c']['unit'])
        self.neurons.d = self.params['d']['value'] * eval(self.params['d']['unit']) 
        self.neurons.u = self.params['u']['value'] * eval(self.params['u']['unit']) 

        return self.neurons
