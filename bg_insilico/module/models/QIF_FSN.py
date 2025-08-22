from brian2 import * 

cubic_current_coeff = 1 * pA / mV**3 

eqs = '''
dv/dt = (k*1*pF/ms/mV*(v - vr) * (v - vt) - u + Isyn + I_ext) / C : volt
du/dt = (1/a) * (int(v <= vb) * (b * (v - vb)**3 * cubic_current_coeff - u) - int(v > vb) * u) : amp

a : second
b : 1
k : 1
vt     : volt
vr     : volt
vb     : volt
th     : volt
c      : volt
C      : farad
d      : amp
I_ext  : amp
cubic_current_coeff : amp / volt**3
'''
