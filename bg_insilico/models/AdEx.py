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
        a      : siemens
        d      : amp
        C      : farad
        I      : amp
        '''

        self.neurons = NeuronGroup(self.N, model=eqs, threshold='v > vt', reset='v = vr; u += d', method='euler')
        self.set_parameters()
        
    def set_parameters(self):
        for param, value in self.params.items():
            setattr(self.neurons, param, value)
            
# R. Naud, N. Marcille, C. Clopath, and W. Gerstner, “Firing patterns in the adaptive exponential integrateand-fire model.,” Biol. Cybern., vol. 99, no. 4–5, pp.335–347, Nov. 2008. 