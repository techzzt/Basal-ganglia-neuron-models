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
        
        self.neurons.vr = self.params['vr']['value'] * eval(self.params['vr']['unit'])
        self.neurons.vt = self.params['vt']['value'] * eval(self.params['vt']['unit'])
        
        # 재현 가능한 seed로 v 초기화 (MSND1 특성: 더 깊은 rest potential)
        np.random.seed(123)  # FSN과 다른 고정 seed
        base_v = self.params['v']['value'] * eval(self.params['v']['unit'])  
        v_noise = np.random.normal(0, 3, self.N) * mV  # MSND1은 더 넓은 분포 ±3mV
        self.neurons.v = base_v + v_noise
        
        self.neurons.th = self.params['th']['value'] * eval(self.params['th']['unit'])
        self.neurons.k = self.params['k']['value'] 
        self.neurons.a = self.params['a']['value'] * eval(self.params['a']['unit'])
        self.neurons.b = self.params['b']['value'] 
        self.neurons.C = self.params['C']['value'] * eval(self.params['C']['unit'])
        self.neurons.c = self.params['c']['value'] * eval(self.params['c']['unit'])
        self.neurons.d = self.params['d']['value'] * eval(self.params['d']['unit']) 
        self.neurons.u = self.params['u']['value'] * eval(self.params['u']['unit'])  # 논문값 유지

        if hasattr(self.neurons, 'g_a'):
            self.neurons.g_a = 0 * nS
        if hasattr(self.neurons, 'g_g'):
            self.neurons.g_g = 0 * nS
        if hasattr(self.neurons, 'g_n'):
            self.neurons.g_n = 0 * nS
            
        for receptor_type, param_list in self.receptor_params.items():
            
            if receptor_type == 'AMPA':
                dominant_params = max(param_list, key=lambda p: p['g0']['value'])
                
                if hasattr(self.neurons, 'tau_AMPA'):
                    self.neurons.tau_AMPA = dominant_params['tau_syn']['value'] * eval(dominant_params['tau_syn']['unit'])
                if hasattr(self.neurons, 'E_AMPA'):
                    self.neurons.E_AMPA = dominant_params['E_rev']['value'] * eval(dominant_params['E_rev']['unit'])
                if hasattr(self.neurons, 'ampa_beta'):
                    self.neurons.ampa_beta = dominant_params.get('beta', {'value': 1.0})['value']
                    
            elif receptor_type == 'GABA':
                dominant_params = max(param_list, key=lambda p: p['g0']['value'])
                
                if hasattr(self.neurons, 'tau_GABA'):
                    self.neurons.tau_GABA = dominant_params['tau_syn']['value'] * eval(dominant_params['tau_syn']['unit'])
                if hasattr(self.neurons, 'E_GABA'):
                    self.neurons.E_GABA = dominant_params['E_rev']['value'] * eval(dominant_params['E_rev']['unit'])
                if hasattr(self.neurons, 'gaba_beta'):
                    self.neurons.gaba_beta = dominant_params.get('beta', {'value': 1.0})['value']
                
            elif receptor_type == 'NMDA':
                dominant_params = max(param_list, key=lambda p: p['g0']['value'])
                
                if hasattr(self.neurons, 'tau_NMDA'):
                    self.neurons.tau_NMDA = dominant_params['tau_syn']['value'] * eval(dominant_params['tau_syn']['unit'])
                if hasattr(self.neurons, 'E_NMDA'):
                    self.neurons.E_NMDA = dominant_params['E_rev']['value'] * eval(dominant_params['E_rev']['unit'])
                if hasattr(self.neurons, 'nmda_beta'):
                    self.neurons.nmda_beta = dominant_params.get('beta', {'value': 1.208})['value']
                if hasattr(self.neurons, 'Mg2'):
                    self.neurons.Mg2 = 1.0

        return self.neurons