from brian2 import *

class NeuronModel:
    def __init__(self, N, params):
        self.N = N
        self.params = params
        self.build_model()

    def build_model(self):
        eqs = '''
        dv/dt = (k*1*pF/ms/mV*(v-vr)*(v-vt)-u*pF+I)/C : volt
        du/dt = a*(b*(v-vr)-u) : volt/second
        a       : 1/second
        b       : 1/second
        c       : volt
        d       : volt/second
        k       : 1
        vr      : volt
        vt      : volt
        vpeak   : volt
        I : amp
        C : farad
        '''

        self.neurons = NeuronGroup(self.N, model=eqs, threshold='v > vt', reset='v = vr; u += d', method='euler')
        self.set_parameters()

    def set_parameters(self):
        for param, value in self.params.items():
            setattr(self.neurons, param, value)