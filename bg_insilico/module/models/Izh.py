from brian2 import *

eqs = '''
dv/dt = (0.04 * (v/mV)**2 + 5 * (v/mV) + 140 - w/pA + (Isyn + I_ext)/pA) * mV/ms : volt
dw/dt = a * (b * (v/mV) * pA - w) : amp

a : second**-1    
b : 1             
c : volt           
d : amp          
vt : volt         
I_ext : amp      
'''

# Reset conditions (to be used with NeuronGroup)
reset = '''
v = c
w += d
'''

threshold = 'v > vt'

