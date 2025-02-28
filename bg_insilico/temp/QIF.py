from brian2 import *

eqs = '''
dv / dt = (k*1*pF/ms/mV*(v-vr)*(v-vt) - u*pF + I - Isyn) / C : volt 
du/dt = a * (b * (v - vr) - u) : volt/second
I : amp
Isyn = I_AMPA + I_GABA : amp
I_AMPA = g_a * (v - E_AMPA) : amp
I_GABA = g_g * (v - E_GABA) : amp
tau_GABA : second
tau_AMPA : second
dg_g/dt = -g_g / tau_GABA : siemens
dg_a/dt = -g_a / tau_AMPA : siemens
E_AMPA : volt
E_GABA : volt
Mg2 : 1
a : 1/second
b : 1
k : 1
vt     : volt
vr     : volt 
th     : volt
c      : volt
C      : farad
d      : volt/second
'''
