from brian2 import *

class NeuronModel:
    def __init__(self, N, params):
        # Parse the parameters from the params dictionary
        self.N = N
        self.params = params
        self.neurons = None

    def create_neurons(self):
        raise NotImplementedError("Subclasses should implement this method.")

class GPeSTNSynapse:
    def __init__(self, GPe, STN, params):
        self.GPe = GPe
        self.STN = STN
        self.params = params

    def create_synapse(self):
        # Create the synapse model where GPe is pre-synaptic and STN is post-synaptic
        
        syn_GPe_STN = Synapses(self.GPe, self.STN, model='''
            g0 : siemens
            g: siemens
            E_GABA : volt
            w : 1
            I_syn_post = w * g * (E_GABA - v_post) : amp (summed)
            ''', 
            
            on_pre='''
            g += g0 
            ''')
        
        syn_GPe_STN.connect(p=0.2) 
        syn_GPe_STN.w = 'rand()' 
        syn_GPe_STN.g0 = self.params['g0']
        # syn_GPe_STN.tau_syn = self.params['tau_syn']
        syn_GPe_STN.E_GABA = self.params['E_GABA']
        # syn_GPe_STN.delay = self.params['delay']

        return syn_GPe_STN