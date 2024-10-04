from brian2 import *

class NeuronModel:
    def __init__(self, N, params):
        # Parse the parameters from the params dictionary
        self.N = N
        self.params = params
        self.neurons = None

    def create_neurons(self):
        raise NotImplementedError("Subclasses should implement this method.")

class Striatum(NeuronModel):
    def create_neurons(self):

        eqs_striatum = '''
        dv/dt = (-g_L*(v-E_L) + I)/(C+ 1e-6 * farad) : volt
        g_L     : siemens
        E_L     : volt
        d       : volt/second
        vr      : volt
        th      : volt
        I : amp
        C : farad
        '''

        
        # Create a dictionary for the namespace
        self.neurons = NeuronGroup(self.N, eqs_striatum, threshold='v > th', reset='v = vr', method='euler')

        return self.neurons