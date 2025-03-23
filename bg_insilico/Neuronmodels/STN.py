from brian2 import *
import importlib
from module.models import AdEx_STN
from module.models import AdEx
import numpy as np
import bisect

from brian2.utils.arrays import calc_repeats
from brian2.utils.logger import get_logger

class NeuronModel:
    def __init__(self, N, params):
        super().__init__(N, params)
        self.neurons = None

    def create_neurons(self):
        raise NotImplementedError("Subclasses should implement this method.")

class STN(NeuronModel):
    def __init__(self, N, params, connections=None): 
        self.N = N
        self.params = params
        self.receptor_params = self.get_receptor_params(connections) if connections else {}
        self.neurons = None
        print(f"[DEBUG] STN receptor_params: {self.receptor_params}")  

    def get_receptor_params(self, connections):
        receptor_params = {}
        for conn_name, conn_data in connections.items():
            if conn_data['post'] == "STN":  
                receptor_params.update(conn_data.get('receptor_params', {}))
        return receptor_params

    def create_neurons(self):
        eqs = AdEx.eqs
        self.neurons = NeuronGroup(
            self.N, eqs, threshold='v > th', reset="u += d; v = vr", method='euler'
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
        self.neurons.I_ext = self.params['I_ext']['value'] * eval(self.params['I_ext']['unit'])


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
