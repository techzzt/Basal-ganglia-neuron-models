from brian2 import *

class NeuronModel:
    def __init__(self, N, params):
        self.N = N
        self.params = params
        self.build_model()

    def build_model(self):
        eqs = '''
        dv_m/dt = (-g_L*(v_m-E_L) + g_L*Delta_T*exp((v_m-V_T)/Delta_T) - w + I)/C : volt
        dw/dt = (a*(v_m-E_L) - w)/tau_w : amp
        I : amp
        g_L      : 1
        Delta_T  : volt
        V_T      : volt
        E_L      : volt
        tau_w    : second
        a        : 1/second
        C        : farad
        '''

        self.neurons = NeuronGroup(self.N, model=eqs, threshold='v_m > V_T', reset='v_m = E_L; w += d', method='euler')
        self.set_parameters()

    def set_parameters(self):
        for param, value in self.params.items():
            setattr(self.neurons, param, value)
