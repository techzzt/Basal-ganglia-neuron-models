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
    def __init__(self, N, params, connections=None):  # ✅ connections 추가
        self.N = N
        self.params = params
        self.receptor_params = self.get_receptor_params(connections) if connections else {}
        self.neurons = None
        print(f"[DEBUG] MSND1 receptor_params: {self.receptor_params}")  # 디버깅용 출력

    def get_receptor_params(self, connections):
        """현재 뉴런이 post-synaptic인 시냅스 파라미터를 connections에서 검색"""
        receptor_params = {}
        for conn_name, conn_data in connections.items():
            if conn_data['post'] == "MSND1":  # ✅ MSND1 뉴런이 post라면 시냅스 정보 가져오기
                receptor_params.update(conn_data.get('receptor_params', {}))
        return receptor_params

    def create_neurons(self):
        eqs = QIF.eqs 
        self.neurons = NeuronGroup(self.N, eqs, threshold='v > th', reset='v = c; u += d', method='euler')

        # 뉴런 파라미터 설정
        self.neurons.vr = self.params['vr']['value'] * eval(self.params['vr']['unit'])
        self.neurons.vt = self.params['vt']['value'] * eval(self.params['vt']['unit'])
        self.neurons.th = self.params['th']['value'] * eval(self.params['th']['unit'])
        self.neurons.k = self.params['k']['value'] 
        self.neurons.a = self.params['a']['value'] / second
        self.neurons.b = self.params['b']['value'] / second
        self.neurons.d = self.params['d']['value'] * volt/second
        self.neurons.C = self.params['C']['value'] * eval(self.params['C']['unit'])
        self.neurons.c = self.params['c']['value'] * eval(self.params['c']['unit'])

        # ✅ self.params가 아니라 self.receptor_params에서 시냅스 정보 가져오기
        rp = self.receptor_params

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

        return self.neurons
