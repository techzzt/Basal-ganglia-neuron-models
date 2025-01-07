from brian2 import *
import importlib
from module.models import AdEx


class NeuronModel:
    def __init__(self, N, params):
        super().__init__(N, params)
        self.neurons = None

    def create_neurons(self):
        raise NotImplementedError("Subclasses should implement this method.")

class GPeT1(NeuronModel):
    def __init__(self, N, params):
        # Parse the parameters from the params dictionary
        self.N = N
        self.params = params
        self.neurons = None

    def create_neurons(self):
        eqs = AdEx.eqs 
            
        self.neurons = NeuronGroup(self.N, eqs, threshold='v > th', reset='v = vr; u += d', method='euler')

        self.neurons.g_L = self.params['g_L']['value'] * eval(self.params['g_L']['unit'])
        self.neurons.E_L = self.params['E_L']['value'] * eval(self.params['E_L']['unit'])
        self.neurons.Delta_T = self.params['Delta_T']['value'] * eval(self.params['Delta_T']['unit'])
        self.neurons.vt = self.params['vt']['value'] * eval(self.params['vt']['unit'])
        self.neurons.vr = self.params['vr']['value'] * eval(self.params['vr']['unit'])
        self.neurons.tau_w = self.params['tau_w']['value'] * eval(self.params['tau_w']['unit'])
        self.neurons.th = self.params['th']['value'] * eval(self.params['th']['unit'])
        self.neurons.a = self.params['a']['value'] * eval(self.params['a']['unit'])
        self.neurons.d = self.params['d']['value'] * eval(self.params['d']['unit'])
        self.neurons.C = self.params['C']['value'] * eval(self.params['C']['unit'])
        
        
        return self.neurons
    