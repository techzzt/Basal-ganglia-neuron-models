from brian2 import *

eqs = '''
dv/dt = (k * (1 * pF/ms/mV) * (v - vr) * (v - vt) - u * pF + I - Isyn) / C : volt
du/dt = int(v <= vb) * (a * b * (v - vb)**3 / (volt**2 * second)) - int(v > vb) * (a * u) : volt/second

I : amp
Isyn = I_AMPA + I_GABA : amp
I_AMPA = g_a * (E_AMPA - v) : amp
I_GABA = g_g * (E_GABA - v) : amp
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
E_L    : volt
vt     : volt
vr     : volt
vb     : volt
th     : volt
c      : volt
C      : farad
d      : volt/second
'''