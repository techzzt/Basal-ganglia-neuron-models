from brian2 import *

eqs = '''
dv / dt = (k*1*pF/ms/mV*(v-vr)*(v-vt) - u*pF + I) / C : volt (unless refractory)
du/dt = a * (b * (v - vr) - u) : volt/second
I = Ispon + Istim + Isyn : amp
Istim   : amp
Ispon   : amp
Isyn: amp
a : 1/second
b : 1/second
k : 1
E_L    : volt
vt     : volt
vr     : volt 
tau_w  : second
th     : volt
c      : volt
C      : farad
d      : volt/second
'''
