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
        self.params['Mg2'] = 1.0 
        
        # ext synapse from STN to GPe
        syn_STN_GPe = Synapses(self.STN, self.GPe, model='''
            g0_a : siemens
            g0_n : siemens
            E_AMPA : volt
            E_NMDA : volt 
            w : 1
            tau_AMPA : second
            tau_NMDA : second
            dg_a/dt = -g_a / tau_AMPA : siemens (clock-driven)
            dg_n/dt = -g_n / tau_NMDA : siemens (clock-driven)
            I_AMPA_syn = w * g_a * (E_AMPA - v) : amp  # Output current variable
            I_NMDA_syn = w * g_n * (E_NMDA - v) / (1 + Mg2 * exp(-0.062 * v_post / mV) / 3.57) : amp
            I_syn_syn = I_AMPA_syn + I_NMDA_syn : amp 
            Mg2 : 1
            ''', 
            on_pre='''
            v_post += w * mV
            ''')

        syn_STN_GPe.connect() 
        syn_STN_GPe.w = 'rand() * 0.7' 
        syn_STN_GPe.g0_n = self.params['g0_n']
        syn_STN_GPe.g0_a = self.params['g0_a']
        syn_STN_GPe.tau_AMPA = self.params['ampa_tau_syn']  
        syn_STN_GPe.tau_NMDA = self.params['nmda_tau_syn']  
        syn_STN_GPe.E_AMPA = self.params['ampa_E_rev']    
        syn_STN_GPe.E_NMDA  = self.params['nmda_E_rev']     
        
        # Inhibitory synapse from EXT to GPe
        syn_Striatum_GPe = Synapses(self.Striatum, self.GPe, model='''
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

        syn_Striatum_GPe.connect() 
        syn_Striatum_GPe.w = 'rand()' 
        syn_Striatum_GPe.g0_g = self.params['g0_g']
        syn_Striatum_GPe.tau_GABA = self.params['gaba_tau_syn']  
        syn_Striatum_GPe.E_GABA = self.params['gaba_E_rev']    
        
        # Excitatory synapse from Cortex to Striatum (MSN)
        syn_Cortex_Striatum = Synapses(self.Cortex, self.Striatum, model='''
            g0_a : siemens
            g0_n : siemens
            E_AMPA : volt
            E_NMDA : volt 
            w : 1
            tau_AMPA : second
            tau_NMDA : second
            dg_a/dt = -g_a / tau_AMPA : siemens (clock-driven)
            dg_n/dt = -g_n / tau_NMDA : siemens (clock-driven)
            I_AMPA_syn = w * g_a * (E_AMPA - v) * s_AMPA_ext: amp  # Output current variable
            I_NMDA_syn = w * g_n * (E_NMDA - v) / (1 + Mg2 * exp(-0.062 * v_post / mV) / 3.57) : amp
            ds_AMPA_ext / dt = - s_AMPA_ext / tau_AMPA : 1
            I_syn_syn = I_AMPA_syn + I_NMDA_syn : amp 
            Mg2 : 1
            ''',
            on_pre='''
            v_post += w * mV
            ''')

        syn_Cortex_Striatum.connect()
        syn_Cortex_Striatum.w = 3.0
        syn_Cortex_Striatum.g0_n = self.params['cs_g0_n']
        syn_Cortex_Striatum.g0_a = self.params['cs_g0_a']
        syn_Cortex_Striatum.tau_AMPA = self.params['cs_ampa_tau_syn']
        syn_Cortex_Striatum.tau_NMDA = self.params['cs_nmda_tau_syn']
        syn_Cortex_Striatum.E_AMPA = self.params['cs_ampa_E_rev']
        syn_Cortex_Striatum.E_NMDA = self.params['cs_nmda_E_rev']
        
        # Excitatory synapse from Cortex to STN
        syn_Cortex_STN = Synapses(self.Cortex, self.STN, model='''
            g0_a : siemens
            g0_n : siemens
            E_AMPA : volt
            E_NMDA : volt 
            w : 1
            tau_AMPA : second
            tau_NMDA : second
            dg_a/dt = -g_a / tau_AMPA : siemens (clock-driven)
            dg_n/dt = -g_n / tau_NMDA : siemens (clock-driven)
            I_AMPA_syn = w * g_a * (E_AMPA - v) : amp  
            I_NMDA_syn = w * g_n * (E_NMDA - v) / (1 + Mg2 * exp(-0.062 * v_post / mV) / 3.57) : amp
            Mg2 : 1
            I_syn_syn = I_AMPA_syn + I_NMDA_syn : amp 
            ''',
            on_pre='''
            v_post += w * mV
            ''')

        syn_Cortex_STN.connect()
        syn_Cortex_STN.w = 'rand()'
        syn_Cortex_STN.g0_n = self.params['csn_g0_n']  
        syn_Cortex_STN.g0_a = self.params['csn_g0_a']  
        syn_Cortex_STN.tau_AMPA = self.params['csn_ampa_tau_syn']  
        syn_Cortex_STN.tau_NMDA = self.params['csn_nmda_tau_syn']  
        syn_Cortex_STN.E_AMPA = self.params['csn_ampa_E_rev']
        syn_Cortex_STN.E_NMDA = self.params['csn_nmda_E_rev']
        
        # Inh synapse from GPe to STN
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

        return syn_STN_GPe, syn_Striatum_GPe, syn_Cortex_Striatum, syn_Cortex_STN, syn_GPe_STN
