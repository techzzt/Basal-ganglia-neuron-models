from brian2 import *

class NeuronModel:
    def __init__(self, N, params):
        # Parse the parameters from the params dictionary
        self.N = N
        self.params = params
        self.neurons = None

    def create_neurons(self):
        raise NotImplementedError("Subclasses should implement this method.")


class STN(NeuronModel):
    def create_neurons(self):
        # Define the STN neuron model based on the params
        eqs_STN = '''
        dv/dt = (-g_L*(v-E_L) + g_L*Delta_T*exp((v-vt)/Delta_T) - u + I + I_syn)/C : volt
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
        I_syn  : amp
        R      : ohm
        '''

        R = 1.0 * ohm

        self.neurons = NeuronGroup(self.N, eqs_STN, threshold='v > th', reset='v = vr; u += d', method='euler')

        # Initialize parameters from the JSON params
        for param, value in self.params.items():
            setattr(self.neurons, param, value)

        return self.neurons