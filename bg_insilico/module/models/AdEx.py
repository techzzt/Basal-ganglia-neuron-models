from brian2 import *

eqs = '''
dv/dt = (-g_L * (v - E_L) + g_L * Delta_T * exp((v - vt) / Delta_T) - z + Isyn + I_ext) / C : volt
dz/dt = (a * (v - E_L) - z) / tau_w : amp

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
