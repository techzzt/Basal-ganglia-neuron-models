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
    def __init__(self, GPe, STN, MSND1, MSND2, SNr, Cortex, params):
        self.GPe = GPe
        self.STN = STN     
        self.MSND1 = MSND1
        self.MSND2 = MSND2
        self.SNr = SNr
        self.Cortex = Cortex
        self.params = params

    def create_synapse(self):
        self.params['Mg2'] = 1.0 
        
        # ext synapse from STN to GPe
        syn_STN_GPe = Synapses(self.STN, self.GPe, model='''
            g0_a : siemens
            E_AMPA : volt
            w : 1
            tau_AMPA : second
            dg_a/dt = -g_a / tau_AMPA : siemens (clock-driven)
            I_AMPA_syn = w * g_a * (E_AMPA - v) : amp  # Output current variable
            I_syn_syn = I_AMPA_syn : amp 
            ''', 
            on_pre='''
            v_post += w * mV; g_a += g0_a
            ''')

        syn_STN_GPe.connect(p = 0.1) 
        syn_STN_GPe.w = 1
        syn_STN_GPe.g0_a = self.params['g0_a']
        syn_STN_GPe.tau_AMPA = self.params['ampa_tau_syn']  
        syn_STN_GPe.E_AMPA = self.params['ampa_E_rev']    
        
        # Inhibitory synapse from EXT to GPe
        syn_MSND2_GPe = Synapses(self.MSND2, self.GPe, model='''
            g0_g : siemens
            E_GABA : volt
            w : 1
            tau_GABA : second
            dg_g/dt = -g_g / tau_GABA : siemens (clock-driven)
            I_GABA_syn = w * g_g * (E_GABA - v) : amp  # Output current variable
            I_syn_syn = I_GABA_syn : amp 
            ''', 
            on_pre='''
            v_post += w * mV; g_g += g0_g
            ''')

        syn_MSND2_GPe.connect(p = 0.2) 
        syn_MSND2_GPe.w = 1
        syn_MSND2_GPe.g0_g = self.params['g0_g']
        syn_MSND2_GPe.tau_GABA = self.params['gaba_tau_syn']  
        syn_MSND2_GPe.E_GABA = self.params['gaba_E_rev']    
    
        # Excitatory synapse from Cortex to MSND1
        syn_Cortex_MSND1 = Synapses(self.Cortex, self.MSND1, model='''
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
            I_NMDA_syn =  - 0.832 * w * g_n * (E_NMDA - v) / (1 + Mg2 * exp(-0.062 * v_post / mV) / 3.57) : amp
            I_syn_syn = I_AMPA_syn + I_NMDA_syn : amp 
            Mg2 : 1
            ''',
            on_pre='''
            v_post += w * mV; g_a += g0_a; g_n += g0_n
            ''')

        syn_Cortex_MSND1.connect(p = 0.8)
        syn_Cortex_MSND1.w = 1
        syn_Cortex_MSND1.g0_n = self.params['cs1_g0_n']
        syn_Cortex_MSND1.g0_a = self.params['cs1_g0_a']
        syn_Cortex_MSND1.tau_AMPA = self.params['cs1_ampa_tau_syn']
        syn_Cortex_MSND1.tau_NMDA = self.params['cs1_nmda_tau_syn']
        syn_Cortex_MSND1.E_AMPA = self.params['cs1_ampa_E_rev']
        syn_Cortex_MSND1.E_NMDA = self.params['cs1_nmda_E_rev']
            
        # Excitatory synapse from Cortex to Striatum (MSN)
        syn_Cortex_MSND2 = Synapses(self.Cortex, self.MSND2, model='''
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
            I_syn_syn = 0.208 * I_AMPA_syn + I_NMDA_syn : amp 
            Mg2 : 1
            ''',
            on_pre='''
            v_post += w * mV; g_a += g0_a; g_n += g0_n
            ''')

        syn_Cortex_MSND2.connect(p = 0.75)
        syn_Cortex_MSND2.w = 1
        syn_Cortex_MSND2.g0_n = self.params['cs_g0_n']
        syn_Cortex_MSND2.g0_a = self.params['cs_g0_a']
        syn_Cortex_MSND2.tau_AMPA = self.params['cs_ampa_tau_syn']
        syn_Cortex_MSND2.tau_NMDA = self.params['cs_nmda_tau_syn']
        syn_Cortex_MSND2.E_AMPA = self.params['cs_ampa_E_rev']
        syn_Cortex_MSND2.E_NMDA = self.params['cs_nmda_E_rev']
        
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
            v_post += w * mV; g_a += g0_a; g_n += g0_n
            ''')

        syn_Cortex_STN.connect(p = 0.75)
        syn_Cortex_STN.w = 1
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
            v_post += w * mV; g_g += g0_g
            ''')

        syn_GPe_STN.connect(p = 0.05)
        syn_GPe_STN.w = 1
        syn_GPe_STN.g0_g = self.params['gsn_g0_g']  
        syn_GPe_STN.tau_GABA = self.params['gsn_gaba_tau_syn']  
        syn_GPe_STN.E_GABA = self.params['gsn_gaba_E_rev']
        
        # Inh synapse from GPe to SNr
        syn_GPe_SNr = Synapses(self.GPe, self.SNr, model='''
            g0_g : siemens
            E_GABA : volt
            w : 1
            tau_GABA : second
            dg_g/dt = -g_g / tau_GABA : siemens (clock-driven)
            I_GABA_syn = w * g_g * (E_GABA - v) : amp  # Output current variable
            I_syn_syn = I_GABA_syn : amp 
            ''', 
            on_pre='''
            v_post += w * mV; g_g += g0_g
            ''')

        syn_GPe_SNr.connect(p = 0.05)
        syn_GPe_SNr.w = 1
        syn_GPe_SNr.g0_g = self.params['gsnr_g0_g']  
        syn_GPe_SNr.tau_GABA = self.params['gsnr_gaba_tau_syn']  
        syn_GPe_SNr.E_GABA = self.params['gsnr_gaba_E_rev']
        
        # Inh synapse from D1 to SNr
        syn_MSND1_SNr = Synapses(self.MSND1, self.SNr, model='''
            g0_g : siemens
            E_GABA : volt
            w : 1
            tau_GABA : second
            dg_g/dt = -g_g / tau_GABA : siemens (clock-driven)
            I_GABA_syn = w * g_g * (E_GABA - v) : amp  # Output current variable
            I_syn_syn = I_GABA_syn : amp 
            ''', 
            on_pre='''
            v_post += w * mV; g_g += g0_g
            ''')

        syn_MSND1_SNr.connect(p = 0.01)
        syn_MSND1_SNr.w = 1
        syn_MSND1_SNr.g0_g = self.params['d1snr_g0_g']  
        syn_MSND1_SNr.tau_GABA = self.params['d1snr_gaba_tau_syn']  
        syn_MSND1_SNr.E_GABA = self.params['d1snr_gaba_E_rev']

        # Excitatory synapse from STN to STr
        syn_STN_SNr = Synapses(self.STN, self.SNr, model='''
            g0_a : siemens
            E_AMPA : volt
            w : 1
            tau_AMPA : second
            dg_a/dt = -g_a / tau_AMPA : siemens (clock-driven)
            I_AMPA_syn = w * g_a * (E_AMPA - v) : amp  
            I_syn_syn = I_AMPA_syn : amp 
            ''',
            on_pre='''
            v_post += w * mV; g_a += g0_a
            ''')

        syn_STN_SNr.connect(p = 0.1)
        syn_STN_SNr.w = 1
        syn_STN_SNr.g0_a = self.params['snsnr_g0_a']  
        syn_STN_SNr.tau_AMPA = self.params['snsnr_ampa_tau_syn']  
        syn_STN_SNr.E_AMPA = self.params['snsnr_ampa_E_rev']
        
        return syn_STN_GPe, syn_MSND2_GPe, syn_Cortex_MSND1, syn_Cortex_MSND2, syn_Cortex_STN, syn_GPe_STN, syn_GPe_SNr, syn_MSND1_SNr, syn_STN_SNr
