# QIF.py
from brian2 import *

eqs = '''
dv/dt = (k * 1 * pF/ms/mV * (v - vr) * (v - vt) - u * pF + I - Isyn) / C : volt 
du/dt = int(v <= vb) * (a * (b * (v - vb)**3 - u)) + int(v > vb) * (-a * u) : volt/second
I : amp
Isyn = I_AMPA + I_NMDA + I_GABA : amp
I_AMPA : amp
I_NMDA : amp
I_GABA : amp
a : 1/second
b : volt**-2/second
k : 1
E_L    : volt
vt     : volt
vr     : volt 
vb     : volt  
tau_w  : second
th     : volt
C      : farad
c      : volt
d      : volt/second
'''
