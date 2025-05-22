from brian2 import *

eqs = '''
dv/dt = (-g_L * (v - E_L) + g_L * Delta_T * exp((v - vt) / Delta_T) - z + I_AMPA + I_NMDA + I_GABA + I_ext) / C : volt
dz/dt = (a * (v - E_L) - z) / tau_w : amp

I_AMPA : amp
I_NMDA : amp
I_GABA : amp

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
