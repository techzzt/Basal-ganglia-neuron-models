from brian2 import *

class NeuronModel:
    def __init__(self, N, params):
        self.N = N
        self.params = params
        self.build_model()

    def build_model(self):
        eqs = '''
        dv/dt = (-g_L*(v-E_L) + I)/C : volt
        g_L     : siemens
        E_L     : volt
        d       : volt/second
        vr      : volt
        th      : volt
        I : amp
        C : farad
        '''

        self.neurons = NeuronGroup(self.N, model=eqs, threshold='v > th', reset='v = vr; th = clip(th + rand()*0.2*mV - 0.1*mV, 0.1*mV, 0.9*mV)', method='euler')
        self.set_parameters()

    def set_parameters(self):
        for param, value in self.params.items():
            setattr(self.neurons, param, value)