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
    def __init__(self, FSN, GPeT1, GPeTA, STN, MSND1, MSND2, SNr, Cortex, params):
        self.FSN = FSN
        self.GPeT1 = GPeT1
        self.GPeTA = GPeTA
        self.STN = STN     
        self.MSND1 = MSND1
        self.MSND2 = MSND2
        self.SNr = SNr
        self.Cortex = Cortex
        self.params = params

    def create_synapse(self):
        self.params['Mg2'] = 1.0 

        # Excitatory synapse from Cortex to FSN
        syn_Cortex_FSN = Synapses(self.Cortex, self.FSN, model='''
            g0_a : siemens
            E_AMPA : volt
            w : 1
            tau_AMPA : second
            dg_a/dt = -g_a / tau_AMPA : siemens (clock-driven)
            I_AMPA_syn = w * g_a * (E_AMPA - v) : amp  # Output current variable
            I_syn_syn = I_AMPA_syn : amp 
            ''',
            on_pre='''
            v_post += w * mV; g_a += g0_a''', delay = self.params['csfs_delay'] * ms)

        syn_Cortex_FSN.connect(p = 0.8) # 바꾸기 
        syn_Cortex_FSN.w = 1
        syn_Cortex_FSN.g0_a = self.params['csfs_g0_a']
        syn_Cortex_FSN.tau_AMPA = self.params['csfs_ampa_tau_syn']
        syn_Cortex_FSN.E_AMPA = self.params['csfs_ampa_E_rev']
        
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
            ''', delay = self.params['cs1_delay'] * ms)

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
            ''', delay = self.params['cs2_delay'] * ms)

        syn_Cortex_MSND2.connect(p = 0.75)
        syn_Cortex_MSND2.w = 1
        syn_Cortex_MSND2.g0_n = self.params['cs2_g0_n']
        syn_Cortex_MSND2.g0_a = self.params['cs2_g0_a']
        syn_Cortex_MSND2.tau_AMPA = self.params['cs2_ampa_tau_syn']
        syn_Cortex_MSND2.tau_NMDA = self.params['cs2_nmda_tau_syn']
        syn_Cortex_MSND2.E_AMPA = self.params['cs2_ampa_E_rev']
        syn_Cortex_MSND2.E_NMDA = self.params['cs2_nmda_E_rev']
        
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
            ''', delay = self.params['csn_delay'] * ms)

        syn_Cortex_STN.connect(p = 0.75)
        syn_Cortex_STN.w = 1
        syn_Cortex_STN.g0_n = self.params['csn_g0_n']  
        syn_Cortex_STN.g0_a = self.params['csn_g0_a']  
        syn_Cortex_STN.tau_AMPA = self.params['csn_ampa_tau_syn']  
        syn_Cortex_STN.tau_NMDA = self.params['csn_nmda_tau_syn']  
        syn_Cortex_STN.E_AMPA = self.params['csn_ampa_E_rev']
        syn_Cortex_STN.E_NMDA = self.params['csn_nmda_E_rev']

        # Inhibitory synapse from FSN to FSN
        syn_FSN_FSN = Synapses(self.FSN, self.FSN, model='''
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
            ''', delay = self.params['fsfs_delay'] * ms)

        syn_FSN_FSN.connect(p = 0.2) # 바꾸기 
        syn_FSN_FSN.w = 1
        syn_FSN_FSN.g0_g = self.params['fsfs_g0_g']
        syn_FSN_FSN.tau_GABA = self.params['fsfs_gaba_tau_syn']  
        syn_FSN_FSN.E_GABA = self.params['fsfs_gaba_E_rev']    
                
        # Inhibitory synapse from FSN to D1
        syn_FSN_MSND1 = Synapses(self.FSN, self.MSND1, model='''
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
            ''', delay = self.params['fsd1_delay'] * ms)

        syn_FSN_MSND1.connect(p = 0.2) # 바꾸기 
        syn_FSN_MSND1.w = 1
        syn_FSN_MSND1.g0_g = self.params['fsd1_g0_g']
        syn_FSN_MSND1.tau_GABA = self.params['fsd1_gaba_tau_syn']  
        syn_FSN_MSND1.E_GABA = self.params['fsd1_gaba_E_rev']    

        # Inhibitory synapse from FSN to D2
        syn_FSN_MSND2 = Synapses(self.FSN, self.MSND2, model='''
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
            ''', delay = self.params['fsd2_delay'] * ms)

        syn_FSN_MSND2.connect(p = 0.2) # 바꾸기 
        syn_FSN_MSND2.w = 1
        syn_FSN_MSND2.g0_g = self.params['fsd2_g0_g']
        syn_FSN_MSND2.tau_GABA = self.params['fsd2_gaba_tau_syn']  
        syn_FSN_MSND2.E_GABA = self.params['fsd2_gaba_E_rev']     
        
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
            ''', delay = self.params['d1snr_delay'] * ms)

        syn_MSND1_SNr.connect(p = 0.01)
        syn_MSND1_SNr.w = 1
        syn_MSND1_SNr.g0_g = self.params['d1snr_g0_g']  
        syn_MSND1_SNr.tau_GABA = self.params['d1snr_gaba_tau_syn']  
        syn_MSND1_SNr.E_GABA = self.params['d1snr_gaba_E_rev']

        syn_MSND1_MSND1 = Synapses(self.MSND1, self.MSND1, model='''
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
            ''', delay = self.params['dd_delay'] * ms)

        syn_MSND1_MSND1.connect(p = 0.01)
        syn_MSND1_MSND1.w = 1
        syn_MSND1_MSND1.g0_g = self.params['d1d1_g0_g']  
        syn_MSND1_MSND1.tau_GABA = self.params['d1d1_gaba_tau_syn']  
        syn_MSND1_MSND1.E_GABA = self.params['d1d1_gaba_E_rev']

        syn_MSND1_MSND2 = Synapses(self.MSND1, self.MSND2, model='''
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
            ''', delay = self.params['dd_delay'] * ms)

        syn_MSND1_MSND2.connect(p = 0.01) # 바꾸기 
        syn_MSND1_MSND2.w = 1
        syn_MSND1_MSND2.g0_g = self.params['d1d2_g0_g']  
        syn_MSND1_MSND2.tau_GABA = self.params['d1d2_gaba_tau_syn']  
        syn_MSND1_MSND2.E_GABA = self.params['d1d2_gaba_E_rev']

        # Inhibitory synapse from D2 to D2
        syn_MSND2_MSND2 = Synapses(self.MSND2, self.MSND2, model='''
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
            ''', delay = self.params['dd_delay'] * ms)

        syn_MSND2_MSND2.connect(p = 0.2) # 바꾸기
        syn_MSND2_MSND2.w = 1
        syn_MSND2_MSND2.g0_g = self.params['d2d2_g0_g']
        syn_MSND2_MSND2.tau_GABA = self.params['d2d2_gaba_tau_syn']  
        syn_MSND2_MSND2.E_GABA = self.params['d2d2_gaba_E_rev']    
    
        # Inhibitory synapse from D2 to D1
        syn_MSND2_MSND1 = Synapses(self.MSND2, self.MSND1, model='''
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
            ''', delay = self.params['dd_delay'] * ms)

        syn_MSND2_MSND1.connect(p = 0.2) # 바꾸기
        syn_MSND2_MSND1.w = 1
        syn_MSND2_MSND1.g0_g = self.params['d2d1_g0_g']
        syn_MSND2_MSND1.tau_GABA = self.params['d2d1_gaba_tau_syn']  
        syn_MSND2_MSND1.E_GABA = self.params['d2d1_gaba_E_rev']   

        # Inhibitory synapse from EXT to GPeT1
        syn_MSND2_GPeT1 = Synapses(self.MSND2, self.GPeT1, model='''
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
            ''', delay = self.params['d2g1_delay'] * ms)

        syn_MSND2_GPeT1.connect(p = 0.2) 
        syn_MSND2_GPeT1.w = 1
        syn_MSND2_GPeT1.g0_g = self.params['d2g1_g0_g']
        syn_MSND2_GPeT1.tau_GABA = self.params['d2g1_gaba_tau_syn']  
        syn_MSND2_GPeT1.E_GABA = self.params['d2g1_gaba_E_rev']    
    
        # ext synapse from STN to GPeT1
        syn_STN_GPeT1 = Synapses(self.STN, self.GPeT1, model='''
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
            ''', delay = self.params['snt1_delay'] * ms)

        syn_STN_GPeT1.connect(p = 0.75) # 바꾸기 
        syn_STN_GPeT1.w = 1
        syn_STN_GPeT1.g0_a = self.params['snt1_g0_a']  
        syn_STN_GPeT1.tau_AMPA = self.params['snt1_ampa_tau_syn']  
        syn_STN_GPeT1.E_AMPA = self.params['snt1_ampa_E_rev']

        # ext synapse from STN to GPeTA
        syn_STN_GPeTA = Synapses(self.STN, self.GPeTA, model='''
            g0_a : siemens
            E_AMPA : volt
            w : 1
            tau_AMPA : second
            dg_a/dt = -g_a / tau_AMPA : siemens (clock-driven)
            I_AMPA_syn = w * g_a * (E_AMPA - v) : amp  
            I_syn_syn = I_AMPA_syn  : amp 
            ''',
            on_pre='''
            v_post += w * mV; g_a += g0_a
            ''')

        syn_STN_GPeTA.connect(p = 0.75) # 바꾸기 
        syn_STN_GPeTA.w = 1
        syn_STN_GPeTA.g0_a = self.params['snta_g0_a']  
        syn_STN_GPeTA.tau_AMPA = self.params['snta_ampa_tau_syn']  
        syn_STN_GPeTA.E_AMPA = self.params['snta_ampa_E_rev']

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
            ''', delay = self.params['snsnr_delay'] * ms)

        syn_STN_SNr.connect(p = 0.1)
        syn_STN_SNr.w = 1
        syn_STN_SNr.g0_a = self.params['snsnr_g0_a']  
        syn_STN_SNr.tau_AMPA = self.params['snsnr_ampa_tau_syn']  
        syn_STN_SNr.E_AMPA = self.params['snsnr_ampa_E_rev']
        
        # Inh synapse from GPeTA to FSN
        syn_GPeT1_FSN = Synapses(self.GPeT1, self.FSN, model='''
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
            ''', delay = self.params['g1fs_delay'] * ms)

        syn_GPeT1_FSN.connect(p = 0.05) # 바꾸기 
        syn_GPeT1_FSN.w = 1
        syn_GPeT1_FSN.g0_g = self.params['g1fs_g0_g']  
        syn_GPeT1_FSN.tau_GABA = self.params['g1fs_gaba_tau_syn']  
        syn_GPeT1_FSN.E_GABA = self.params['g1fs_gaba_E_rev']

        # Inh synapse from GPeT1 to STN
        syn_GPeT1_STN = Synapses(self.GPeT1, self.STN, model='''
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
            ''', delay = self.params['g1sn_delay'] * ms)

        syn_GPeT1_STN.connect(p = 0.05)
        syn_GPeT1_STN.w = 1
        syn_GPeT1_STN.g0_g = self.params['g1sn_g0_g']  
        syn_GPeT1_STN.tau_GABA = self.params['g1sn_gaba_tau_syn']  
        syn_GPeT1_STN.E_GABA = self.params['g1sn_gaba_E_rev']
        
        # Inh synapse from GPeT1 to SNr
        syn_GPeT1_SNr = Synapses(self.GPeT1, self.SNr, model='''
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
            ''', delay = self.params['g1snr_delay'] * ms)

        syn_GPeT1_SNr.connect(p = 0.05)
        syn_GPeT1_SNr.w = 1
        syn_GPeT1_SNr.g0_g = self.params['g1snr_g0_g']  
        syn_GPeT1_SNr.tau_GABA = self.params['g1snr_gaba_tau_syn']  
        syn_GPeT1_SNr.E_GABA = self.params['g1snr_gaba_E_rev']
        
        # Inh synapse from GPeT1 to T1
        syn_GPeT1_GPeT1 = Synapses(self.GPeT1, self.GPeT1, model='''
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
            ''', delay = self.params['g1g1_delay'] * ms)

        syn_GPeT1_GPeT1.connect(p = 0.05) # 바꾸기 
        syn_GPeT1_GPeT1.w = 1
        syn_GPeT1_GPeT1.g0_g = self.params['g1g1_g0_g']  
        syn_GPeT1_GPeT1.tau_GABA = self.params['g1g1_gaba_tau_syn']  
        syn_GPeT1_GPeT1.E_GABA = self.params['g1g1_gaba_E_rev']          
         
        # Inh synapse from GPeT1 to TA
        syn_GPeT1_GPeTA = Synapses(self.GPeT1, self.GPeTA, model='''
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
            ''', delay = self.params['g1ga_delay'] * ms)

        syn_GPeT1_GPeTA.connect(p = 0.05) # 바꾸기 
        syn_GPeT1_GPeTA.w = 1
        syn_GPeT1_GPeTA.g0_g = self.params['g1ga_g0_g']  
        syn_GPeT1_GPeTA.tau_GABA = self.params['g1ga_gaba_tau_syn']  
        syn_GPeT1_GPeTA.E_GABA = self.params['g1ga_gaba_E_rev']

        # Inh synapse from GPeT1 to TA
        syn_GPeTA_GPeT1 = Synapses(self.GPeTA, self.GPeT1, model='''
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
            ''', delay = self.params['gag1_delay'] * ms)

        syn_GPeTA_GPeT1.connect(p = 0.05) # 바꾸기 
        syn_GPeTA_GPeT1.w = 1
        syn_GPeTA_GPeT1.g0_g = self.params['gag1_g0_g']  
        syn_GPeTA_GPeT1.tau_GABA = self.params['gag1_gaba_tau_syn']  
        syn_GPeTA_GPeT1.E_GABA = self.params['gag1_gaba_E_rev']
        
        # Inh synapse from GPeTA to TA
        syn_GPeTA_GPeTA = Synapses(self.GPeTA, self.GPeTA, model='''
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
            ''', delay = self.params['gag1_delay'] * ms)

        syn_GPeTA_GPeTA.connect(p = 0.05) # 바꾸기 
        syn_GPeTA_GPeTA.w = 1
        syn_GPeTA_GPeTA.g0_g = self.params['gaga_g0_g']  
        syn_GPeTA_GPeTA.tau_GABA = self.params['gaga_gaba_tau_syn']  
        syn_GPeTA_GPeTA.E_GABA = self.params['gaga_gaba_E_rev']

        # Inh synapse from GPeTA to FSN
        syn_GPeTA_FSN = Synapses(self.GPeTA, self.FSN, model='''
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
            ''', delay = self.params['gafs_delay'] * ms)

        syn_GPeTA_FSN.connect(p = 0.05) # 바꾸기 
        syn_GPeTA_FSN.w = 1
        syn_GPeTA_FSN.g0_g = self.params['gafs_g0_g']  
        syn_GPeTA_FSN.tau_GABA = self.params['gafs_gaba_tau_syn']  
        syn_GPeTA_FSN.E_GABA = self.params['gafs_gaba_E_rev']

        # Inh synapse from GPeTA to D2
        syn_GPeTA_MSND1 = Synapses(self.GPeTA, self.MSND1, model='''
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
            ''', delay = self.params['gad1_delay'] * ms)

        syn_GPeTA_MSND1.connect(p = 0.05) # 바꾸기 
        syn_GPeTA_MSND1.w = 1
        syn_GPeTA_MSND1.g0_g = self.params['gad1_g0_g']  
        syn_GPeTA_MSND1.tau_GABA = self.params['gad1_gaba_tau_syn']  
        syn_GPeTA_MSND1.E_GABA = self.params['gad1_gaba_E_rev']

        # Inh synapse from GPeTA to D2
        syn_GPeTA_MSND2 = Synapses(self.GPeTA, self.MSND2, model='''
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
            ''', delay = self.params['gad2_delay'] * ms)

        syn_GPeTA_MSND2.connect(p = 0.05) # 바꾸기 
        syn_GPeTA_MSND2.w = 1
        syn_GPeTA_MSND2.g0_g = self.params['gad2_g0_g']  
        syn_GPeTA_MSND2.tau_GABA = self.params['gad2_gaba_tau_syn']  
        syn_GPeTA_MSND2.E_GABA = self.params['gad2_gaba_E_rev']


        return syn_Cortex_FSN, syn_Cortex_MSND1, syn_Cortex_MSND2, syn_Cortex_STN, syn_FSN_FSN, syn_FSN_MSND1, syn_FSN_MSND2, syn_MSND1_SNr, syn_MSND1_MSND1, syn_MSND1_MSND2, syn_MSND2_MSND2, syn_MSND2_MSND1, syn_MSND2_GPeT1, syn_STN_GPeT1, syn_STN_GPeTA, syn_STN_SNr, syn_GPeT1_FSN, syn_GPeT1_STN, syn_GPeT1_SNr, syn_GPeT1_GPeT1, syn_GPeT1_GPeTA, syn_GPeTA_GPeT1, syn_GPeTA_GPeTA, syn_GPeTA_FSN, syn_GPeTA_MSND1, syn_GPeTA_MSND2