from brian2 import *

class NeuronModel:
    def __init__(self, N, params):
        self.N = N
        self.params = params
        self.build_model()

    def build_model(self):
        eqs = '''
        dv/dt = (-g_L*1*pF/ms/mV*(v-E_L) + I)/C : volt
        g_L     : 1/second
        E_L     : 1/second
        d       : volt/second
        vr      : volt
        vt      : volt
        I : amp
        C : farad
        '''

        self.neurons = NeuronGroup(self.N, model=eqs, threshold='v > vt', reset='v = vr; u += d', method='euler')
        self.set_parameters()

    def set_parameters(self):
        for param, value in self.params.items():
            setattr(self.neurons, param, value)