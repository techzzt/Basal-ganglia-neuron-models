from brian2 import *

class NeuronModel:
    def __init__(self, N, params):
        # Parse the parameters from the params dictionary
        self.N = N
        self.params = params
        self.neurons = None

    def create_neurons(self):
        raise NotImplementedError("Subclasses should implement this method.")

class Synapse:
    def __init__(self, GPe, STN, params):
        self.GPe = GPe
        self.STN = STN
        self.params = params

    def create_synapse(self):
        
        syn_GPe_STN = Synapses(self.GPe, self.STN, model='''
            g0_g : siemens
            E_GABA : volt
            w : 1
            tau_GABA : second
            dg_g/dt = -g_g / tau_GABA : siemens (clock-driven)
            I_GABA_syn = w * g_g * (E_GABA - v) : amp  # Output current variable
            I_syn_syn = I_GABA_syn : amp 
            ''', 
            on_pre='''
            v_post += w * mV
            ''')
        
        syn_GPe_STN.connect()
        syn_GPe_STN.w = 'rand()'
        syn_GPe_STN.g0_g = self.params['gsn_g0_g']  
        syn_GPe_STN.tau_GABA = self.params['gsn_gaba_tau_syn']  
        syn_GPe_STN.E_GABA = self.params['gsn_gaba_E_rev']


        return syn_GPe_STN