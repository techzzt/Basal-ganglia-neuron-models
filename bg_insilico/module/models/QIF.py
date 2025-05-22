from brian2 import *

eqs = '''
dv/dt = (k*1*pF/ms/mV*(v-vr)*(v-vt) - u + I_AMPA + I_NMDA + I_GABA) / C : volt 
du/dt = (1/a) * (b * pA * (v - vr) / mV - u) : amp

I_AMPA : amp
I_NMDA : amp
I_GABA : amp

a : second
b : 1
k : 1
vt     : volt
vr     : volt 
th     : volt
c      : volt
C      : farad
d      : amp
'''