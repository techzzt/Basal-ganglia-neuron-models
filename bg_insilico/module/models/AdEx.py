from brian2 import *

eqs = '''
dv/dt = (-g_L * (v - E_L) + g_L * Delta_T * exp((v - vt) / Delta_T) - z + Isyn + I_ext) / C : volt
dz/dt = (a * (v - E_L) - z) / tau_w : amp
Isyn = I_AMPA + I_NMDA + I_GABA : amp
I_AMPA = ampa_beta * g_a * (E_AMPA - v) : amp 
I_GABA = gaba_beta * g_g * (E_GABA - v) : amp
I_NMDA = nmda_beta * g_n * (E_NMDA - v) / (1 + Mg2 * exp(-0.062 * v / mV) / 3.57) : amp 

tau_GABA : second
tau_AMPA : second
tau_NMDA : second
dg_g/dt = -g_g / tau_GABA : siemens
dg_a/dt = -g_a / tau_AMPA : siemens
dg_n/dt = -g_n / tau_NMDA : siemens

E_AMPA : volt
E_GABA : volt
E_NMDA : volt
ampa_beta: 1
gaba_beta: 1
nmda_beta: 1
Mg2 : 1

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
I_ext  : amp
'''
