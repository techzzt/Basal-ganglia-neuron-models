# QIF.py
from brian2 import *

eqs = '''
dv/dt = (k * 1 * pF/ms/mV * (v - vr) * (v - vt) - u * pF + I) / C : volt (unless refractory)
du/dt = int(v <= vb) * (a * (b * (vb - v)**3 - u)) + int(v > vb) * (-a * u) : volt/second
I = Ispon + Istim + Isyn : amp
Istim   : amp
Ispon   : amp
Isyn    : amp
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
