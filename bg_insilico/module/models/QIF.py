from brian2 import *

eqs = '''
dv/dt = (k*1*pF/ms/mV*(v-vr)*(v-vt) - u + Isyn) / C : volt 
du/dt = (1/a) * (b * pA * (v - vr) / mV - u) : amp

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