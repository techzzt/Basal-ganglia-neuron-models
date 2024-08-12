from brian2 import *

class NeuronModel:
    def __init__(self, N, params):
        self.N = N
        self.params = params
        self.build_model()

    def build_model(self):
        eqs = '''
        dv/dt = (0.04/ms/mV)*v**2+(5/ms)*v+140*mV/ms-u+I : volt
        du/dt = a*(b*v-u)                                : volt/second
        a       : 1/second
        b       : 1/second
        d       : volt/second
        vr      : volt
        I       : volt/second
        '''

        self.neurons = NeuronGroup(self.N, model=eqs, threshold='v >= 30*mV', reset='v = vr; u += d', method='euler')
        self.set_parameters()

    def set_parameters(self):
        for param, value in self.params.items():
            setattr(self.neurons, param, value)


# https://groups.google.com/g/briansupport/c/39rcKe5xdsE