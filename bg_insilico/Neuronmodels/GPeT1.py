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
    def __init__(self, N, params, receptor_params=None):
        self.N = N
        self.params = params
        self.receptor_params = receptor_params
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
        
        if 'receptor_params' in self.params:
            rp = self.params['receptor_params']
            # AMPA parameters
            if 'AMPA' in rp:
                self.neurons.E_AMPA = rp['AMPA']['E_rev']['value'] * eval(rp['AMPA']['E_rev']['unit'])
                self.neurons.tau_AMPA = rp['AMPA']['tau_syn']['value'] * eval(rp['AMPA']['tau_syn']['unit'])
                self.neurons.ampa_beta = rp['AMPA']['beta']['value']
            else:
                self.neurons.E_AMPA = 0 * mV
                self.neurons.tau_AMPA = 1 * ms
                self.neurons.ampa_beta = 0

            # NMDA parameters
            if 'NMDA' in rp:
                self.neurons.E_NMDA = rp['NMDA']['E_rev']['value'] * eval(rp['NMDA']['E_rev']['unit'])
                self.neurons.tau_NMDA = rp['NMDA']['tau_syn']['value'] * eval(rp['NMDA']['tau_syn']['unit'])
                self.neurons.nmda_beta = rp['NMDA']['beta']['value']
            else:
                self.neurons.E_NMDA = 0 * mV
                self.neurons.tau_NMDA = 1 * ms
                self.neurons.nmda_beta = 0

            # GABA parameters
            if 'GABA' in rp:
                self.neurons.E_GABA = rp['GABA']['E_rev']['value'] * eval(rp['GABA']['E_rev']['unit'])
                self.neurons.tau_GABA = rp['GABA']['tau_syn']['value'] * eval(rp['GABA']['tau_syn']['unit'])
                self.neurons.gaba_beta = rp['GABA']['beta']['value']
            else:
                self.neurons.E_GABA = 0 * mV
                self.neurons.tau_GABA = 1 * ms
                self.neurons.gaba_beta = 0
        else:
            # defaults (convert to zero)
            self.neurons.E_AMPA = 0 * mV
            self.neurons.tau_AMPA = 1 * ms
            self.neurons.ampa_beta = 0
            self.neurons.E_NMDA = 0 * mV
            self.neurons.tau_NMDA = 1 * ms
            self.neurons.nmda_beta = 0
            self.neurons.E_GABA = 0 * mV
            self.neurons.tau_GABA = 1 * ms
            self.neurons.gaba_beta = 0
        
        return self.neurons
    
    