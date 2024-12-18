from brian2 import *

# Define the equations for the neuron populations
eqs = '''
dv/dt = (k * 1 * pF/ms/mV * (v - vr) * (v - vt) - u * pF + I) / C : volt (unless refractory)
du/dt = int(v <= vb) * (a * (b * (vb - v)**3 - u)) + int(v > vb) * (-a * u) : volt/second
I = Ispon + Istim + Isyn : amp
Istim   : amp
Ispon   : amp
Isyn = I_AMPA_FSN + I_NMDA + I_GABA_FSN: amp
I_GABA_FSN  : amp
I_AMPA_FSN : amp
I_NMDA : amp
a : 1/second
b : volt**-2/second
k : 1
E_L    : volt
vt     : volt
vr     : volt 
vb     : volt  
tau_w  : second
th     : volt
C      : farad
c      : volt
d      : volt/second
'''

class NeuronModel:
    def __init__(self, N, params):
        # Parse the parameters from the params dictionary
        super().__init__(N, params)
        self.neurons = None

    def create_neurons(self):
        raise NotImplementedError("Subclasses should implement this method.")

# Class for the FSN neuron model
class FSN(NeuronModel):
    def __init__(self, N, params):
        self.N = N
        self.params = params
        self.neurons = None

    def create_neurons(self):
        self.neurons = NeuronGroup(
            self.N, eqs, threshold='v > th', reset='v = c; u += d', method='euler'
        )

        # Set parameters using the dictionary
        self.neurons.vr = self.params['vr']['value'] * eval(self.params['vr']['unit'])
        self.neurons.vt = self.params['vt']['value'] * eval(self.params['vt']['unit'])
        self.neurons.th = self.params['th']['value'] * eval(self.params['th']['unit'])
        self.neurons.vb = self.params['vb']['value'] * eval(self.params['vb']['unit'])
        self.neurons.k = self.params['k']['value']
        self.neurons.a = self.params['a']['value'] / second
        self.neurons.b = self.params['b']['value'] / (volt**2 * second)  # 단위 변경
        self.neurons.d = self.params['d']['value']
        self.neurons.C = self.params['C']['value'] * eval(self.params['C']['unit'])
        self.neurons.c = self.params['c']['value'] * eval(self.params['c']['unit'])

        return self.neurons
