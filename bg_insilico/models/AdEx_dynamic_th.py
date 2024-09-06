from brian2 import *

class NeuronModel:
    def __init__(self, N, params):
        self.N = N
        self.params = params
        self.build_model()

    def build_model(self):
        eqs = '''
        dv/dt = (-g_L*(v-E_L) + g_L*Delta_T*exp((v-vt)/Delta_T) - u + I)/C : volt
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
        '''

        self.neurons = NeuronGroup(self.N, model=eqs, threshold='v > th', reset='v = vr; u += d; th = clip(th + rand()*0.2*mV - 0.1*mV, 0.1*mV, 0.9*mV)', method='euler')
        self.set_parameters()
        
    def set_parameters(self):
        for param, value in self.params.items():
            setattr(self.neurons, param, value)
        