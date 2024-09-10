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
        dv/dt = (-g_L*(v-E_L) + g_L*Delta_T*exp((v-vt)/Delta_T) - u + I)/C + sigma*sqrt(2/tau)*xi: volt
        du/dt = (a*(v-E_L) - u)/tau_w : amp
        g_L    : siemens
        E_L    : volt
        Delta_T: volt
        vt     : volt
        vr     : volt 
        tau_w  : second
        th     : volt
        a      : siemens
        d      : amp
        C      : farad
        I      : amp
        sigma   : volt
        tau     : second
        '''

        self.neurons = NeuronGroup(self.N, model=eqs, threshold='v > th', reset='v = vr; u += d', method='euler')
        self.set_parameters()
        
    def set_parameters(self):
        for param, value in self.params.items():
            setattr(self.neurons, param, value)
        self.neurons.sigma = self.sigma
        self.neurons.tau = self.tau