from brian2 import *

class NeuronModel:
    def __init__(self, N, params):
        # Parse the parameters from the params dictionary
        self.N = N
        self.params = params
        self.neurons = None

    def create_neurons(self):
        raise NotImplementedError("Subclasses should implement this method.")

class GPeModel(NeuronModel):
    def create_neurons(self):
        # Define the GPe neuron model based on the params
        eqs_GPe = '''
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
        self.neurons = NeuronGroup(self.N, eqs_GPe, threshold='v > th', reset='v = vr; u += d', method='euler')

        # Initialize parameters from the JSON params
        for param, value in self.params.items():
            setattr(self.neurons, param, value)

        return self.neurons

class STNModel(NeuronModel):
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
        '''
        self.neurons = NeuronGroup(self.N, eqs_STN, threshold='v > th', reset='v = vr; u += d', method='euler')

        # Initialize parameters from the JSON params
        for param, value in self.params.items():
            setattr(self.neurons, param, value)

        return self.neurons

class GPeSTNSynapse:
    def __init__(self, GPe, STN, params):
        self.GPe = GPe
        self.STN = STN
        self.params = params

    def create_synapse(self):
        # Create the synapse model between GPe and STN
        
        syn_GPe_STN = Synapses(self.GPe, self.STN, model='''
            w : siemens
            tau_syn : second
            E_GABA : volt
            dg/dt = -g/tau_syn : siemens (clock-driven)
            I_syn_post = g * (E_GABA - v_post) : amp (summed)
            ''', 
            on_pre='''
            g += w  # Increase conductance on spike 
            ''')
        
        syn_GPe_STN.connect(p=0.2)  # Connect with probability 0.2
        syn_GPe_STN.w = self.params['w']
        syn_GPe_STN.tau_syn = self.params['tau_syn']
        syn_GPe_STN.E_GABA = self.params['E_GABA']
        syn_GPe_STN.delay = self.params['delay']

        return syn_GPe_STN

