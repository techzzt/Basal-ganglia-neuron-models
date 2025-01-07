import importlib
from brian2 import *
from module.models import QIF

class NeuronModel:
    def __init__(self, N, params):
        super().__init__(N, params)
        self.neurons = None

    def create_neurons(self):
        raise NotImplementedError("Subclasses should implement this method.")

class MSND1(NeuronModel):
    def __init__(self, N, params):
        self.N = N
        self.params = params
        self.neurons = None

    def create_neurons(self):
        eqs = QIF.eqs 
        self.neurons = NeuronGroup(self.N, eqs, threshold='v > th', reset='v = c; u += d', method='euler')

        # Initialize parameters with their proper units
        self.neurons.vr = self.params['vr']['value'] * eval(self.params['vr']['unit'])
        self.neurons.vt = self.params['vt']['value'] * eval(self.params['vt']['unit'])
        self.neurons.th = self.params['th']['value'] * eval(self.params['th']['unit'])
        self.neurons.k = self.params['k']['value'] 
        self.neurons.a = self.params['a']['value'] / second
        self.neurons.b = self.params['b']['value'] * (Hz) 
        self.neurons.d = self.params['d']['value'] * mV / ms
        self.neurons.C = self.params['C']['value'] * eval(self.params['C']['unit'])
        self.neurons.c = self.params['c']['value'] * eval(self.params['c']['unit'])

        return self.neurons

    