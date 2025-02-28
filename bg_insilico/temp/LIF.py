from brian2 import *


eqs = '''
dv/dt = (-g_L*(v-E_L) + I)/C : volt (unless refractory)
g_L     : siemens
E_L     : volt
d       : volt/second
vr      : volt
th      : volt
I : amp
C : farad
'''