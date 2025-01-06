from brian2 import *
from module.models.synapse_models import SynapseBase

class GPe_STN_inh_ext_dop_dep(SynapseBase):
    def __init__(self, neurons, params):
        super().__init__(neurons, params)
        self.params['Mg2'] = {'value': 1.0, 'unit': '1'}
        self.equations() 

    def equations(self):
        self.equations = {
            'AMPA': '''
                g0_a : siemens
                E_AMPA : volt
                w : 1
                ampa_beta : 1
                tau_AMPA : second
                dg_a/dt = -g_a / tau_AMPA : siemens (clock-driven)
                I_AMPA_syn = ampa_beta * w * g_a * (E_AMPA - v) : amp
            ''',
            'NMDA': '''
                g0_n : siemens
                Mg2: 1
                w: 1
                E_NMDA : volt 
                nmda_beta: 1
                tau_NMDA : second
                dg_n/dt = -g_n / tau_NMDA : siemens (clock-driven)
                I_NMDA_syn = nmda_beta * w * g_n * (E_NMDA - v) / (1 + Mg2 * exp(-0.062 * v / mV) / 3.57) : amp
            ''',
            'GABA': '''
                g0_g : siemens
                E_GABA : volt
                w : 1
                tau_GABA : second
                gaba_beta : 1
                dg_g/dt = -g_g / tau_GABA : siemens (clock-driven)
                I_GABA_syn = gaba_beta * w * g_g * (E_GABA - v) : amp
            '''
        }

    def create_synapse(self):
        synapses = []

        # Excitatory synapse from Cortex to FSN
        syn_Cortex_FSN = Synapses(self.Cortex_FSN, self.FSN, model=self.equations['AMPA'],
            on_pre='''
            v_post += w * mV; g_a += g0_a''',         
            delay=self._get_param('csfs_delay'))
        syn_Cortex_FSN.connect(p = 0.75)
        syn_Cortex_FSN.w = 1
        syn_Cortex_FSN.g0_a = self._get_param('csfs_g0_a')
        syn_Cortex_FSN.tau_AMPA = self._get_param('csfs_ampa_tau_syn')
        syn_Cortex_FSN.E_AMPA = self._get_param('csfs_ampa_E_rev')
        synapses.append(syn_Cortex_FSN)

        syn_Cortex_MSND1_ampa = Synapses(
            self.Cortex_MSND1, self.MSND1, 
            model=self.equations['AMPA'], 
            on_pre='''
            v_post += w * mV; g_a += g0_a;
            ''',
        )
        syn_Cortex_MSND1_ampa.connect(p=0.75)
        syn_Cortex_MSND1_ampa.w = 1
        # Assign AMPA parameters after connecting
        syn_Cortex_MSND1_ampa.ampa_beta = self._get_param('cs1_ampa_beta')
        syn_Cortex_MSND1_ampa.g0_a = self._get_param('cs1_g0_a')
        syn_Cortex_MSND1_ampa.tau_AMPA = self._get_param('cs1_ampa_tau_syn')
        syn_Cortex_MSND1_ampa.E_AMPA = self._get_param('cs1_ampa_E_rev')

        synapses.append(syn_Cortex_MSND1_ampa)

        # Create NMDA synapse
        syn_Cortex_MSND1_nmda = Synapses(
            self.Cortex_MSND1, self.MSND1, 
            model=self.equations['NMDA'], 
            on_pre='''
            v_post += w * mV; g_n += g0_n;
            ''',
        )
        syn_Cortex_MSND1_nmda.connect(p=0.75)

        # Assign NMDA parameters after connecting
        syn_Cortex_MSND1_nmda.nmda_beta = self._get_param('cs1_nmda_beta')
        syn_Cortex_MSND1_nmda.g0_n = self._get_param('cs1_g0_n')
        syn_Cortex_MSND1_nmda.tau_NMDA = self._get_param('cs1_nmda_tau_syn')
        syn_Cortex_MSND1_nmda.E_NMDA = self._get_param('cs1_nmda_E_rev')

        synapses.append(syn_Cortex_MSND1_nmda)
            
        # Cortex to MSND2 Synapse (AMPA + NMDA)
        syn_Cortex_MSND2_ampa = Synapses(
            self.Cortex_MSND2, self.MSND2, 
            model=self.equations['AMPA'], 
            on_pre='''
            v_post += w * mV; g_a += g0_a;
            ''', 
        )
        syn_Cortex_MSND2_ampa.connect(p=0.75)
        syn_Cortex_MSND2_ampa.ampa_beta = self._get_param('cs2_ampa_beta')
        syn_Cortex_MSND2_ampa.g0_a = self._get_param('cs2_g0_a')
        syn_Cortex_MSND2_ampa.tau_AMPA = self._get_param('cs2_ampa_tau_syn')
        syn_Cortex_MSND2_ampa.E_AMPA = self._get_param('cs2_ampa_E_rev')
        synapses.append(syn_Cortex_MSND2_ampa)

        syn_Cortex_MSND2_nmda = Synapses(
            self.Cortex_MSND2, self.MSND2, 
            model=self.equations['NMDA'], 
            on_pre='''
            v_post += w * mV; g_n += g0_n;
            ''', 
        )
        syn_Cortex_MSND2_nmda.connect(p=0.75)
        syn_Cortex_MSND2_nmda.nmda_beta = self._get_param('cs2_nmda_beta')
        syn_Cortex_MSND2_nmda.g0_n = self._get_param('cs2_g0_n')
        syn_Cortex_MSND2_nmda.tau_NMDA = self._get_param('cs2_nmda_tau_syn')
        syn_Cortex_MSND2_nmda.E_NMDA = self._get_param('cs2_nmda_E_rev')
        synapses.append(syn_Cortex_MSND2_nmda)

        # Cortex to STN Synapse
        syn_Cortex_STN_ampa = Synapses(
            self.Cortex_STN, self.STN, 
            model=self.equations['AMPA'], 
            on_pre='''
            v_post += w * mV; g_a += g0_a;
            ''', 
            delay=self._get_param('csn_delay')
        )
        syn_Cortex_STN_ampa.connect(p=0.75)
        syn_Cortex_STN_ampa.ampa_beta = self._get_param('csn_ampa_beta')
        syn_Cortex_STN_ampa.g0_a = self._get_param('csn_g0_a')
        syn_Cortex_STN_ampa.tau_AMPA = self._get_param('csn_ampa_tau_syn')
        syn_Cortex_STN_ampa.E_AMPA = self._get_param('csn_ampa_E_rev')
        synapses.append(syn_Cortex_STN_ampa)

        syn_Cortex_STN_nmda = Synapses(
            self.Cortex_STN, self.STN, 
            model=self.equations['NMDA'], 
            on_pre='''
            v_post += w * mV; g_n += g0_n;
            ''', 
            delay=self._get_param('csn_delay')
        )
        syn_Cortex_STN_nmda.connect(p=0.75)
        syn_Cortex_STN_nmda.nmda_beta = self._get_param('csn_nmda_beta')
        syn_Cortex_STN_nmda.g0_n = self._get_param('csn_g0_n')
        syn_Cortex_STN_nmda.tau_NMDA = self._get_param('csn_nmda_tau_syn')
        syn_Cortex_STN_nmda.E_NMDA = self._get_param('csn_nmda_E_rev')
        synapses.append(syn_Cortex_STN_nmda)

        # FSN to FSN Synapse
        syn_FSN_FSN = Synapses(
            self.FSN, self.FSN, 
            model=self.equations['GABA'], 
            on_pre='''
            g_g += g0_g;
            ''', 
            delay=self._get_param('fsfs_delay')
        )
        syn_FSN_FSN.connect(p=0.74)
        syn_FSN_FSN.g0_g = self._get_param('fsfs_g0_g')
        syn_FSN_FSN.gaba_beta = self._get_param('fsfs_gaba_beta')
        syn_FSN_FSN.tau_GABA = self._get_param('fsfs_gaba_tau_syn')
        syn_FSN_FSN.E_GABA = self._get_param('fsfs_gaba_E_rev')
        synapses.append(syn_FSN_FSN)

        # FSN to MSND1 Synapse
        syn_FSN_MSND1 = Synapses(
            self.FSN, self.MSND1, 
            model=self.equations['GABA'], 
            on_pre='''
            g_g += g0_g;
            ''', 
            delay=self._get_param('fsd1_delay')
        )
        syn_FSN_MSND1.connect(p=0.27)
        syn_FSN_MSND1.g0_g = self._get_param('fsd1_g0_g')
        syn_FSN_MSND1.tau_GABA = self._get_param('fsd1_gaba_tau_syn')
        syn_FSN_MSND1.E_GABA = self._get_param('fsd1_gaba_E_rev')
        synapses.append(syn_FSN_MSND1)

        # FSN to MSND2 Synapse
        syn_FSN_MSND2 = Synapses(
            self.FSN, self.MSND2, 
            model=self.equations['GABA'], 
            on_pre='''
            g_g += g0_g;
            ''', 
            delay=self._get_param('fsd2_delay')
        )
        syn_FSN_MSND2.connect(p=0.18)
        syn_FSN_MSND2.g0_g = self._get_param('fsd2_g0_g')
        syn_FSN_MSND2.tau_GABA = self._get_param('fsd2_gaba_tau_syn')
        syn_FSN_MSND2.E_GABA = self._get_param('fsd2_gaba_E_rev')
        synapses.append(syn_FSN_MSND2)

        # MSND1 to SNr Synapse
        syn_MSND1_SNr = Synapses(
            self.MSND1, self.SNr, 
            model=self.equations['GABA'], 
            on_pre='''
            g_g += g0_g;
            ''', 
            delay=self._get_param('d1snr_delay')
        )
        syn_MSND1_SNr.connect(p=0.1)
        syn_MSND1_SNr.g0_g = self._get_param('d1snr_g0_g')
        syn_MSND1_SNr.gaba_beta = self._get_param('d1snr_gaba_beta')
        syn_MSND1_SNr.tau_GABA = self._get_param('d1snr_gaba_tau_syn')
        syn_MSND1_SNr.E_GABA = self._get_param('d1snr_gaba_E_rev')
        synapses.append(syn_MSND1_SNr)

        # GABAergic synapse from MSND1 to MSND1
        syn_MSND1_MSND1 = Synapses(self.MSND1, self.MSND1, model=self.equations['GABA'], 
            on_pre='''g_g += g0_g''', delay=self._get_param('dd_delay'))
        syn_MSND1_MSND1.connect(p=0.18)
        syn_MSND1_MSND1.w = 1
        syn_MSND1_MSND1.g0_g = self._get_param('d1d1_g0_g')
        syn_MSND1_MSND1.gaba_beta = self._get_param('d1d1_gaba_beta')
        syn_MSND1_MSND1.tau_GABA = self._get_param('d1d1_gaba_tau_syn')
        syn_MSND1_MSND1.E_GABA = self._get_param('d1d1_gaba_E_rev')
        synapses.append(syn_MSND1_MSND1)

        # GABAergic synapse from MSND1 to MSND2
        syn_MSND1_MSND2 = Synapses(self.MSND1, self.MSND2, model=self.equations['GABA'], 
            on_pre='''g_g += g0_g''', delay=self._get_param('dd_delay'))
        syn_MSND1_MSND2.connect(p=0.03)
        syn_MSND1_MSND2.w = 1
        syn_MSND1_MSND2.g0_g = self._get_param('d1d2_g0_g')
        syn_MSND1_MSND2.gaba_beta = self._get_param('d1d2_gaba_beta')
        syn_MSND1_MSND2.tau_GABA = self._get_param('d1d2_gaba_tau_syn')
        syn_MSND1_MSND2.E_GABA = self._get_param('d1d2_gaba_E_rev')
        synapses.append(syn_MSND1_MSND2)

        # GABAergic synapse from MSND2 to MSND2
        syn_MSND2_MSND2 = Synapses(self.MSND2, self.MSND2, model=self.equations['GABA'], 
            on_pre='''g_g += g0_g''', delay=self._get_param('dd_delay'))
        syn_MSND2_MSND2.connect(p=0.18)
        syn_MSND2_MSND2.w = 1
        syn_MSND2_MSND2.g0_g = self._get_param('d2d2_g0_g')
        syn_MSND2_MSND2.gaba_beta = self._get_param('d2d2_gaba_beta')
        syn_MSND2_MSND2.tau_GABA = self._get_param('d2d2_gaba_tau_syn')
        syn_MSND2_MSND2.E_GABA = self._get_param('d2d2_gaba_E_rev')
        synapses.append(syn_MSND2_MSND2)

        # GABAergic synapse from MSND2 to MSND1
        syn_MSND2_MSND1 = Synapses(self.MSND2, self.MSND1, model=self.equations['GABA'], 
            on_pre='''g_g += g0_g''', delay=self._get_param('dd_delay'))
        syn_MSND2_MSND1.connect(p=0.14)
        syn_MSND2_MSND1.w = 1
        syn_MSND2_MSND1.g0_g = self._get_param('d2d1_g0_g')
        syn_MSND2_MSND1.gaba_beta = self._get_param('d2d1_gaba_beta')
        syn_MSND2_MSND1.tau_GABA = self._get_param('d2d1_gaba_tau_syn')
        syn_MSND2_MSND1.E_GABA = self._get_param('d2d1_gaba_E_rev')
        synapses.append(syn_MSND2_MSND1)

        # GABAergic synapse from MSND2 to GPeT1
        syn_MSND2_GPeT1 = Synapses(self.MSND2, self.GPeT1, model=self.equations['GABA'], 
            on_pre='''g_g += g0_g''', delay=self._get_param('d2g1_delay'))
        syn_MSND2_GPeT1.connect(p=0.2)
        syn_MSND2_GPeT1.w = 1
        syn_MSND2_GPeT1.g0_g = self._get_param('d2g1_g0_g')
        syn_MSND2_GPeT1.gaba_beta = self._get_param('d2g1_gaba_beta')
        syn_MSND2_GPeT1.tau_GABA = self._get_param('d2g1_gaba_tau_syn')
        syn_MSND2_GPeT1.E_GABA = self._get_param('d2g1_gaba_E_rev')
        synapses.append(syn_MSND2_GPeT1)
        
       # Synapse from STN to GPeT1 (AMPA model)
        syn_STN_GPeT1 = Synapses(self.STN, self.GPeT1, model=self.equations['AMPA'],
            on_pre='''
            v_post += w * mV; g_a += g0_a
            ''', delay=self._get_param('snt1_delay'))
        syn_STN_GPeT1.connect(p=0.1)
        syn_STN_GPeT1.w = 1
        syn_STN_GPeT1.g0_a = self._get_param('snt1_g0_a')
        syn_STN_GPeT1.ampa_beta = self._get_param('snt1_ampa_beta')
        syn_STN_GPeT1.tau_AMPA = self._get_param('snt1_ampa_tau_syn')
        syn_STN_GPeT1.E_AMPA = self._get_param('snt1_ampa_E_rev')
        synapses.append(syn_STN_GPeT1)

        # Synapse from STN to GPeTA (AMPA model)
        syn_STN_GPeTA = Synapses(self.STN, self.GPeTA, model=self.equations['AMPA'],
            on_pre='''
            v_post += w * mV; g_a += g0_a
            ''')
        syn_STN_GPeTA.connect(p=0.1)
        syn_STN_GPeTA.w = 1
        syn_STN_GPeTA.ampa_beta = self._get_param('snta_ampa_beta')
        syn_STN_GPeTA.g0_a = self._get_param('snta_g0_a')
        syn_STN_GPeTA.tau_AMPA = self._get_param('snta_ampa_tau_syn')
        syn_STN_GPeTA.E_AMPA = self._get_param('snta_ampa_E_rev')
        synapses.append(syn_STN_GPeTA)

        # Synapse from STN to SNr (AMPA model)
        syn_STN_SNr = Synapses(self.STN, self.SNr, model=self.equations['AMPA'],
            on_pre='''
            v_post += w * mV; g_a += g0_a
            ''', delay=self._get_param('snsnr_delay'))
        syn_STN_SNr.connect(p=0.1)
        syn_STN_SNr.w = 1
        syn_STN_SNr.g0_a = self._get_param('snsnr_g0_a')
        syn_STN_SNr.ampa_beta = self._get_param('snsnr_ampa_beta')
        syn_STN_SNr.tau_AMPA = self._get_param('snsnr_ampa_tau_syn')
        syn_STN_SNr.E_AMPA = self._get_param('snsnr_ampa_E_rev')
        synapses.append(syn_STN_SNr)

        # Inhibitory synapse from GPeT1 to FSN (GABA model)
        syn_GPeT1_FSN = Synapses(self.GPeT1, self.FSN, model=self.equations['GABA'],
            on_pre='''
            g_g += g0_g
            ''', delay=self._get_param('g1fs_delay'))
        syn_GPeT1_FSN.connect(p=0.01)
        syn_GPeT1_FSN.w = 1
        syn_GPeT1_FSN.g0_g = self._get_param('g1fs_g0_g')
        syn_GPeT1_FSN.gaba_beta = self._get_param('g1fs_gaba_beta')
        syn_GPeT1_FSN.tau_GABA = self._get_param('g1fs_gaba_tau_syn')
        syn_GPeT1_FSN.E_GABA = self._get_param('g1fs_gaba_E_rev')
        synapses.append(syn_GPeT1_FSN)

        # Inhibitory synapse from GPeT1 to STN (GABA model)
        syn_GPeT1_STN = Synapses(self.GPeT1, self.STN, model=self.equations['GABA'],
            on_pre='''
            g_g += g0_g
            ''', delay=self._get_param('g1sn_delay'))
        syn_GPeT1_STN.connect(p=0.03)
        syn_GPeT1_STN.w = 1
        syn_GPeT1_STN.g0_g = self._get_param('g1sn_g0_g')
        syn_GPeT1_STN.gaba_beta = self._get_param('g1sn_gaba_beta')
        syn_GPeT1_STN.tau_GABA = self._get_param('g1sn_gaba_tau_syn')
        syn_GPeT1_STN.E_GABA = self._get_param('g1sn_gaba_E_rev')
        synapses.append(syn_GPeT1_STN)

        syn_GPeT1_SNr = Synapses(self.GPeT1, self.SNr, model=self.equations['GABA'],
            on_pre='''
            g_g += g0_g
            ''', delay=self._get_param('g1snr_delay'))
        syn_GPeT1_SNr.connect(p=0.03)
        syn_GPeT1_SNr.w = 1
        syn_GPeT1_SNr.g0_g = self._get_param('g1snr_g0_g')
        syn_GPeT1_SNr.gaba_beta = self._get_param('g1snr_gaba_beta')
        syn_GPeT1_SNr.tau_GABA = self._get_param('g1snr_gaba_tau_syn')
        syn_GPeT1_SNr.E_GABA = self._get_param('g1snr_gaba_E_rev')
        synapses.append(syn_GPeT1_SNr)

        syn_GPeT1_GPeT1 = Synapses(self.GPeT1, self.GPeT1, model=self.equations['GABA'],
            on_pre='''
            g_g += g0_g
            ''', delay=self._get_param('g1g1_delay'))
        syn_GPeT1_GPeT1.connect(p=0.02)
        syn_GPeT1_GPeT1.w = 1
        syn_GPeT1_GPeT1.g0_g = self._get_param('g1g1_g0_g')
        syn_GPeT1_GPeT1.gaba_beta = self._get_param('g1g1_gaba_beta')
        syn_GPeT1_GPeT1.tau_GABA = self._get_param('g1g1_gaba_tau_syn')
        syn_GPeT1_GPeT1.E_GABA = self._get_param('g1g1_gaba_E_rev')
        synapses.append(syn_GPeT1_GPeT1)

        syn_GPeT1_GPeTA = Synapses(self.GPeT1, self.GPeTA, model=self.equations['GABA'],
            on_pre='''
            g_g += g0_g
            ''', delay=self._get_param('g1ga_delay'))
        syn_GPeT1_GPeTA.connect(p=0.02)
        syn_GPeT1_GPeTA.w = 1
        syn_GPeT1_GPeTA.g0_g = self._get_param('g1ga_g0_g')
        syn_GPeT1_GPeTA.gaba_beta = self._get_param('g1ga_gaba_beta')
        syn_GPeT1_GPeTA.tau_GABA = self._get_param('g1ga_gaba_tau_syn')
        syn_GPeT1_GPeTA.E_GABA = self._get_param('g1ga_gaba_E_rev')
        synapses.append(syn_GPeT1_GPeTA)

        syn_GPeTA_GPeT1 = Synapses(self.GPeTA, self.GPeT1, model=self.equations['GABA'],
            on_pre='''
            g_g += g0_g
            ''', delay=self._get_param('gag1_delay'))
        syn_GPeTA_GPeT1.connect(p=0.05)
        syn_GPeTA_GPeT1.w = 1
        syn_GPeTA_GPeT1.g0_g = self._get_param('gag1_g0_g')
        syn_GPeTA_GPeT1.gaba_beta = self._get_param('gag1_gaba_beta')
        syn_GPeTA_GPeT1.tau_GABA = self._get_param('gag1_gaba_tau_syn')
        syn_GPeTA_GPeT1.E_GABA = self._get_param('gag1_gaba_E_rev')
        synapses.append(syn_GPeTA_GPeT1)

        syn_GPeTA_GPeTA = Synapses(self.GPeTA, self.GPeTA, model=self.equations['GABA'],
            on_pre='''
            g_g += g0_g
            ''', delay=self._get_param('gaga_delay'))
        syn_GPeTA_GPeTA.connect(p=0.02)
        syn_GPeTA_GPeTA.w = 1
        syn_GPeTA_GPeTA.g0_g = self._get_param('gaga_g0_g')
        syn_GPeTA_GPeTA.gaba_beta = self._get_param('gaga_gaba_beta')
        syn_GPeTA_GPeTA.tau_GABA = self._get_param('gaga_gaba_tau_syn')
        syn_GPeTA_GPeTA.E_GABA = self._get_param('gaga_gaba_E_rev')
        synapses.append(syn_GPeTA_GPeTA)

        syn_GPeTA_FSN = Synapses(self.GPeTA, self.FSN, model=self.equations['GABA'],
            on_pre='''
            g_g += g0_g
            ''', delay=self._get_param('gafs_delay'))
        syn_GPeTA_FSN.connect(p=0.03)
        syn_GPeTA_FSN.w = 1
        syn_GPeTA_FSN.g0_g = self._get_param('gafs_g0_g')
        syn_GPeTA_FSN.gaba_beta = self._get_param('gafs_gaba_beta')
        syn_GPeTA_FSN.tau_GABA = self._get_param('gafs_gaba_tau_syn')
        syn_GPeTA_FSN.E_GABA = self._get_param('gafs_gaba_E_rev')
        synapses.append(syn_GPeTA_FSN)

        syn_GPeTA_MSND1 = Synapses(self.GPeTA, self.MSND1, model=self.equations['GABA'],
            on_pre='''
            g_g += g0_g
            ''', delay=self._get_param('gad1_delay'))
        syn_GPeTA_MSND1.connect(p=0.03)
        syn_GPeTA_MSND1.w = 1
        syn_GPeTA_MSND1.g0_g = self._get_param('gad1_g0_g')
        syn_GPeTA_MSND1.gaba_beta = self._get_param('gad1_gaba_beta')
        syn_GPeTA_MSND1.tau_GABA = self._get_param('gad1_gaba_tau_syn')
        syn_GPeTA_MSND1.E_GABA = self._get_param('gad1_gaba_E_rev')
        synapses.append(syn_GPeTA_MSND1)

        syn_GPeTA_MSND2 = Synapses(self.GPeTA, self.MSND2, model=self.equations['GABA'],
            on_pre='''
            g_g += g0_g
            ''', delay=self._get_param('gad2_delay'))
        syn_GPeTA_MSND2.connect(p=0.03)
        syn_GPeTA_MSND2.w = 1
        syn_GPeTA_MSND2.g0_g = self._get_param('gad2_g0_g')
        syn_GPeTA_MSND1.gaba_beta = self._get_param('gad2_gaba_beta')
        syn_GPeTA_MSND2.tau_GABA = self._get_param('gad2_gaba_tau_syn')
        syn_GPeTA_MSND2.E_GABA = self._get_param('gad2_gaba_E_rev')
        synapses.append(syn_GPeTA_MSND2)

        return synapses