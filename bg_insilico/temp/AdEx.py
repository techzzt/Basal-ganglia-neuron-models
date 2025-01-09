from brian2 import *

eqs = '''
dv/dt = (-g_L * (v - E_L) + g_L * Delta_T * exp((v - vt) / Delta_T) - u + I) / C : volt
du/dt = (a * (v - E_L) - u) / tau_w : amp
I = Ispon + Istim + Isyn : amp
Istim   : amp
Ispon   : amp
Isyn : amp
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
'''