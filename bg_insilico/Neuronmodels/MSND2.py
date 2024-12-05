from brian2 import *

# Define the equations for the neuron populations
eqs = '''
dv / dt = (k*1*pF/ms/mV*(v-vr)*(v-vt) - u*pF + I) / C : volt (unless refractory)
du/dt = a * (b * (v - vr) - u) : volt/second
I = Ispon + Istim + Isyn : amp
Istim   : amp
Ispon   : amp
Isyn = I_AMPA + I_NMDA + I_GABA_MSND2: amp
I_GABA_MSND2  : amp
I_AMPA : amp
I_NMDA : amp
a : 1/second
b : 1/second
k : 1
E_L    : volt
vt     : volt
vr     : volt 
tau_w  : second
th     : volt
C      : farad
d      : volt/second
'''


class NeuronModel:
    def __init__(self, N, params):
        super().__init__(N, params)
        self.neurons = None

    def create_neurons(self):
        raise NotImplementedError("Subclasses should implement this method.")

class MSND2(NeuronModel):
    def __init__(self, N, params):
        # Parse the parameters from the params dictionary
        self.N = N
        self.params = params
        self.neurons = None

    def create_neurons(self):
        # Define the neuron model based on the type
        self.neurons = NeuronGroup(self.N, eqs, threshold='v > th', reset='v = vr; u += d', method='euler')

        # Initialize parameters with their proper units
        self.neurons.v = self.params['vr']['value'] * eval(self.params['vr']['unit'])
        self.neurons.vr = self.params['vr']['value'] * eval(self.params['vr']['unit'])
        self.neurons.vt = self.params['vt']['value'] * eval(self.params['vt']['unit'])
        self.neurons.th = self.params['th']['value'] * eval(self.params['th']['unit'])
        self.neurons.k = self.params['k']['value'] 
        self.neurons.a = self.params['a']['value'] / second
        self.neurons.b = self.params['b']['value'] * (Hz) 
        self.neurons.d = self.params['d']['value'] * mV / ms
        self.neurons.C = self.params['C']['value'] * eval(self.params['C']['unit'])

        return self.neurons

    