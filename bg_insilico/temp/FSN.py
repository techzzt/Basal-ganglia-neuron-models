import importlib
from brian2 import *

from module.models import QIF_FSN
from module.models import QIF
from module.models import LIF

class NeuronModel:
    def __init__(self, N, params):
        super().__init__(N, params)
        self.neurons = None

    def create_neurons(self):
        raise NotImplementedError("Subclasses should implement this method.")

class FSN(NeuronModel):
    def __init__(self, N, params):
        self.N = N
        self.params = params
        self.neurons = None

    def create_neurons(self):
        eqs = QIF_FSN.eqs 

        self.neurons = NeuronGroup(
            self.N, eqs, threshold='v > th', reset='v = vr', method='euler'
        )

        self.neurons.vr = self.params['vr']['value'] * eval(self.params['vr']['unit'])
        self.neurons.vt = self.params['vt']['value'] * eval(self.params['vt']['unit'])
        self.neurons.th = self.params['th']['value'] * eval(self.params['th']['unit'])
        self.neurons.k = self.params['k']['value'] 
        self.neurons.a = self.params['a']['value'] / second
        self.neurons.b = self.params['b']['value'] 
        self.neurons.d = self.params['d']['value'] * volt/second
        self.neurons.C = self.params['C']['value'] * eval(self.params['C']['unit'])
        self.neurons.c = self.params['c']['value'] * eval(self.params['c']['unit'])
        self.neurons.vb = self.params['vb']['value'] * eval(self.params['vb']['unit']) 
        
        self.neurons.E_AMPA = 0 * mV
        self.neurons.E_GABA = -74 * mV
        self.neurons.tau_GABA = 6 * ms
        self.neurons.tau_AMPA = 12 * ms
        self.neurons.gaba_beta = 0.746
    
        return self.neurons