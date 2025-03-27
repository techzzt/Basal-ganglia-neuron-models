from brian2 import *

eqs = '''
dv/dt = (k*1*pF/ms/mV*(v-vr)*(v-vt) - u*pF + Isyn) / C : volt 
du/dt = a * (b * ((v - vb) ** 3/volt ** 2) - u) : volt/second

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

a : 1/second
b : 1/second
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