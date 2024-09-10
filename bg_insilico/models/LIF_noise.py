from brian2 import *

class NeuronModel:
    def __init__(self, N, params, sigma = 3 * mV, tau = 1 * ms):
        self.N = N
        self.params = params
        self.sigma = sigma
        self.tau = tau
        self.build_model()

    def build_model(self):
        eqs = '''
        dv/dt = (-g_L*(v-E_L) + I)/C + sigma*sqrt(2/tau)*xi : volt
        g_L     : siemens
        E_L     : volt
        vr      : volt
        th      : volt
        I       : amp
        C       : farad
        sigma   : volt
        tau     : second
        '''

        self.neurons = NeuronGroup(self.N, model=eqs, threshold='v > th', reset='v = vr', method='euler')
        self.set_parameters()

    def set_parameters(self):
        for param, value in self.params.items():
            setattr(self.neurons, param, value)
        self.neurons.sigma = self.sigma
        self.neurons.tau = self.tau
