from brian2 import *

# Define the equations for the neuron populations
eqs_E = '''
dv / dt = (- g_L * (v - E_L) + g_L * Delta_T * exp((v - vt) / Delta_T) - u +I - I_syn) / C : volt (unless refractory)
du/dt = (a * (v - E_L) - u) / tau_w : amp

I_syn = I_AMPA + I_NMDA : amp

I_AMPA : amp
I_NMDA : amp

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

eqs_I = '''
dv / dt = (- g_L * (v - E_L) + g_L * Delta_T * exp((v - vt) / Delta_T) - u + I - I_syn) / C : volt (unless refractory)
du/dt = (a * (v - E_L) - u) / tau_w : amp

I_syn = I_GABA : amp

I_GABA_rec : amp
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

class NeuronModel:
    def __init__(self, N, params):
        # Parse the parameters from the params dictionary
        self.N = N
        self.params = params
        self.neurons = None

    def create_neurons(self):
        raise NotImplementedError("Subclasses should implement this method.")

class STN(NeuronModel):
    def __init__(self, N, params, neuron_type='E'):
        # Parse the parameters from the params dictionary
        self.N = N
        self.params = params
        self.neuron_type = neuron_type  # 'E' for excitatory, 'I' for inhibitory
        self.neurons = None

    def create_neurons(self):
        if self.neuron_type == 'E':
            eqs = eqs_E
        else:
            eqs = eqs_I

        # Create the neuron group using the appropriate equations
        self.neurons = NeuronGroup(self.N, eqs, threshold='v > th', reset='v = vr; u += d', method='euler')
        for param, value in self.params.items():
            setattr(self.neurons, param, value)
        
        return self.neurons