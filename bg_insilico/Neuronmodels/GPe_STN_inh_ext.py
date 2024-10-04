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
    def __init__(self, GPe, STN, Striatum, Cortex, params):
        self.GPe = GPe
        self.STN = STN     
        self.Striatum = Striatum
        self.Cortex = Cortex
        self.params = params

    def create_synapse(self):
        # Inhibitory synapse from GPe to STN
        syn_GPe_STN = Synapses(self.GPe, self.STN, model='''
            g0 : siemens
            E_GABA : volt
            w : 1
            tau_syn : second
            I_syn_GPe_STN : amp  # Renamed synaptic variable
            dg/dt = -g / tau_syn : siemens (clock-driven)
            I_out_GPe_STN = w * g * (E_GABA - v_post) : amp  # Renamed output variable
            ''', 
            on_pre='''
            g += g0 
            ''')
        
        syn_GPe_STN.connect(p=0.2) 
        syn_GPe_STN.w = 'rand()' 
        syn_GPe_STN.g0 = self.params['g0']
        syn_GPe_STN.tau_syn = self.params['tau_syn']
        syn_GPe_STN.E_GABA = self.params['E_GABA']
        
        # Excitatory synapse from Cortex to Striatum
        syn_Cortex_Striatum = Synapses(self.Cortex, self.Striatum, model='''
            g0_striatum : siemens
            E_AMPA : volt
            w : 1
            tau_AMPA : second
            I_syn_Cortex_Striatum : amp  # Renamed synaptic variable
            dg/dt = -g / tau_AMPA : siemens (clock-driven)
            I_out_Cortex_Striatum = w * g * (E_AMPA - v_post) : amp  # Renamed output variable
            ''', 
            on_pre='''
            g += g0_striatum
            ''')

        syn_Cortex_Striatum.connect(p=0.5)
        syn_Cortex_Striatum.w = 'rand() * 0.1'
        syn_Cortex_Striatum.g0_striatum = self.params['striatum_g0']
        syn_Cortex_Striatum.E_AMPA = self.params['striatum_ampa_E_rev']
        syn_Cortex_Striatum.tau_AMPA = self.params['striatum_ampa_tau_syn']
        
        # Excitatory synapse from Cortex to STN
        syn_Cortex_STN = Synapses(self.Cortex, self.STN, model='''
            g0_stn : siemens
            E_AMPA : volt
            w : 1
            tau_AMPA : second
            I_syn_Cortex_STN : amp  # Renamed synaptic variable
            dg/dt = -g / tau_AMPA : siemens (clock-driven)
            I_out_Cortex_STN = w * g * (E_AMPA - v_post) : amp  # Renamed output variable
            ''', 
            on_pre='''
            g += g0_stn
            ''')

        syn_Cortex_STN.connect(p=0.5)
        syn_Cortex_STN.w = 'rand() * 0.1'
        syn_Cortex_STN.g0_stn = self.params['stn_ampa_g0']
        syn_Cortex_STN.E_AMPA = self.params['stn_ampa_E_rev']
        syn_Cortex_STN.tau_AMPA = self.params['stn_ampa_tau_syn']

        return syn_GPe_STN, syn_Cortex_Striatum, syn_Cortex_STN

"""
class GPeSTNSynapse:
    def __init__(self, GPe, STN, Striatum, Cortex, params):
        self.GPe = GPe
        self.STN = STN     
        self.Striatum = Striatum
        self.Cortex = Cortex
        self.params = params

    def create_synapse(self):
        # Create the synapse model where GPe is pre-synaptic and STN is post-synaptic
        
       # GPe → STN Synapse
        syn_GPe_STN = Synapses(self.GPe, self.STN, model='''
            g0 : siemens
            E_GABA : volt
            w : 1
            tau_syn : secondx
            I_syn_post = w * g * (E_GABA - v_post) : amp (summed)
            dg/dt = -g / tau_syn : siemens (clock-driven)
            ''', 
            on_pre='''
            g += g0 
            ''')
        
        syn_GPe_STN.connect(p=0.75) # T1: 0.75, TA: 0.25
        syn_GPe_STN.w = 'rand()' 
        syn_GPe_STN.g0 = self.params['g0']
        syn_GPe_STN.E_GABA = self.params['E_GABA']
        syn_GPe_STN.tau_syn = self.params['tau_syn']

        # Striatum → GPe Synapse
        syn_Str_GPe = Synapses(self.Striatum, self.GPe, model='''
            g0 : siemens
            E_GABA : volt
            w : 1
            tau_inh : second
            I_inh_post = w * g * (E_GABA - v_post) : amp (summed)
            dg/dt = -g / tau_inh : siemens (clock-driven)
            ''', 
            on_pre='''
            g += g0
            ''')
        
        syn_Str_GPe.connect(p=0.25)
        syn_Str_GPe.w = 'rand()'
        syn_Str_GPe.g0 = self.params['striatum_g0']
        syn_Str_GPe.E_GABA = self.params['striatum_E_GABA']
        syn_Str_GPe.tau_inh = self.params['tau_syn_Str']

        # Cortex → STN Synapse
        syn_Cortex_STN = Synapses(self.Cortex, self.STN, model='''
            g0 : siemens
            E_AMPA : volt
            w : 1
            tau_ext : second
            I_ext_post = w * g * (E_AMPA - v_post) : amp (summed)
            dg/dt = -g / tau_ext : siemens (clock-driven)
            ''', 
            on_pre='''
            g += g0
            ''')
        
        syn_Cortex_STN.connect(p=0.4)
        syn_Cortex_STN.w = 'rand()'
        syn_Cortex_STN.g0 = self.params['cortex_g0']
        syn_Cortex_STN.E_AMPA = self.params['cortex_E_AMPA']
        syn_Cortex_STN.tau_ext = self.params['tau_syn_Cortex']

        return syn_GPe_STN, syn_Str_GPe, syn_Cortex_STN
"""