from brian2 import *

# Define the equations for the neuron populations
eqs_E = '''
dv / dt = (k*1*pF/ms/mV*(v-vr)*(v-vt) - u*pF + I - I_syn) / C : volt (unless refractory)
du/dt = a * (b * (v - vr) - u) : volt/second

I_syn = I_AMPA + I_NMDA : amp

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
I      : amp
d       : volt/second
'''

eqs_I = '''
dv / dt = (k*1*pF/ms/mV*(v-vr)*(v-vt) - u*pF + I - I_syn) / C : volt (unless refractory)
du/dt = a * (b * (v - vr) - u) / tau_w : amp

I_syn = I_GABA : amp

I_GABA  : amp
a : 1/second
b : volt/second
k : 1
E_L    : volt
vt     : volt
vr     : volt 
tau_w  : second
th     : volt
C      : farad
I      : amp
d       : volt/second
'''

class NeuronModel:
    def __init__(self, N, params):
        # Parse the parameters from the params dictionary
        self.N = N
        self.params = params
        self.neurons = None

    def create_neurons(self):
        raise NotImplementedError("Subclasses should implement this method.")

class Striatum(NeuronModel):
    def __init__(self, N, params, neuron_type='E'):
        # Parse the parameters from the params dictionary
        self.N = N
        self.params = params
        self.neuron_type = neuron_type  # 'E' for excitatory, 'I' for inhibitory
        self.neurons = None

    def create_neurons(self):
        # Define the neuron model based on the type
        if self.neuron_type == 'E':
            eqs = eqs_E
        else:
            eqs = eqs_I

        # Create the neuron group using euler integration
        self.neurons = NeuronGroup(self.N, eqs, threshold='v >= th', reset='v = vr; u += d', method='euler')

        # Initialize parameters with their proper units
        self.neurons.v = self.params['vr'] 
        self.neurons.vt = self.params['vt'] 
        self.neurons.th = self.params['th'] 
        self.neurons.C = self.params['C'] 
        self.neurons.a = self.params['a'] / second  # Parameter a in 1/second
        self.neurons.b = self.params['b'] * Hz # Parameter b in 1/second
        self.neurons.I = self.params['I'] 

        return self.neurons

    