from brian2 import *

def generate_synapse_model(max_receptors=5):

    base_eqs = '''
    dv/dt = (-g_L * (v - E_L) + g_L * Delta_T * exp((v - vt) / Delta_T) - z + I_syn_total + I_ext) / C : volt
    dz/dt = (a * (v - E_L) - z) / tau_w : amp
    '''
    
    receptors = ['AMPA', 'NMDA', 'GABA']
    
    for receptor in receptors:
        for i in range(max_receptors):
            base_eqs += f'I_syn_{receptor}_{i} : amp\n'
    
    for receptor in receptors:
        terms = [f'I_syn_{receptor}_{i}' for i in range(max_receptors)]
        base_eqs += f'I_syn_{receptor} = {" + ".join(terms)} : amp\n'
    
    base_eqs += 'I_syn_total = I_syn_AMPA + I_syn_NMDA + I_syn_GABA : amp\n'
    
    base_eqs += '''
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
    I_ext  : amp
    '''
    
    return base_eqs

eqs = generate_synapse_model(max_receptors=5)
