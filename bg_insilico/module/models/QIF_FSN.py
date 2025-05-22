from brian2 import * 

cubic_current_coeff = 1 * pA / mV**3 

eqs = '''
dv/dt = (k*1*pF/ms/mV*(v - vr) * (v - vt) - u + I_AMPA + I_NMDA + I_GABA) / C : volt
du/dt = (1/a) * (int(v <= vb) * (b * (v - vb)**3 * cubic_current_coeff - u) - int(v > vb) * u) : amp

I_AMPA : amp
I_NMDA : amp
I_GABA : amp

a : second
b : 1
k : 1
E_L    : volt
vt     : volt
vr     : volt
vb     : volt
th     : volt
c      : volt
C      : farad
d      : amp
cubic_current_coeff : amp / volt**3
'''
