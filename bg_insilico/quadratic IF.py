from brian2 import *

class NeuronModel:
    def __init__(self, N, params):
        self.N = N
        self.params = params
        self.build_model()

    def build_model(self):
        eqs = '''
        dv/dt = (K*1*pF/ms/mV*(v-VR)*(v-vt)-u*pF+I)/C : volt
        du/dt = a*(b*(v-VR)-u) : volt/second
        VR = vr*(1+KAPA*Dop1) : volt
        K = k*(1-ALPHA*Dop2) : 1
        a       : 1/second
        b       : 1/second
        c       : volt
        d       : volt/second
        k       : 1
        vr      : volt
        vt      : volt
        vpeak   : volt
        I : amp
        Dop1      : 1
        Dop2      : 1
        KAPA      : 1
        ALPHA     : 1
        C : farad
        '''

        self.neurons = NeuronGroup(self.N, model=eqs, threshold='v > vpeak', reset='v = c; u += d', method='euler')
        self.set_parameters()

    def set_parameters(self):
        for param, value in self.params.items():
            setattr(self.neurons, param, value)