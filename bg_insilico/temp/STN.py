from brian2 import *
import importlib
from module.models import AdEx

class NeuronModel:
    def __init__(self, N, params):
        super().__init__(N, params)
        self.neurons = None

    def create_neurons(self):
        raise NotImplementedError("Subclasses should implement this method.")
    

class STN(NeuronModel):
    def __init__(self, N, params):
        self.N = N
        self.params = params
        self.neurons = None

    def create_neurons(self):
        eqs = AdEx.eqs
        """
        reset = '''
        v = vr + clip(u - 15*mV, 20*mV, inf*mV);
        u += d
        '''
        """
        reset = '''
        v = vr;
        u += d
        '''
        self.neurons = NeuronGroup(
            self.N, eqs, threshold='v > th', reset=reset, method='euler'
        )

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
       
        self.neurons.E_AMPA = 0 * mV
        self.neurons.tau_AMPA = 4 * ms
        self.neurons.ampa_beta = 1.09
        
        self.neurons.E_NMDA = 0 * mV
        self.neurons.tau_NMDA = 160 * ms
        self.neurons.nmda_beta = 1
        self.neurons.tau_GABA = 1 * ms
        
        return self.neurons