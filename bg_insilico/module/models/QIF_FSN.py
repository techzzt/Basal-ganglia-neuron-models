from brian2 import * 

cubic_current_coeff = 1 * pA / mV**3 

def generate_synapse_model(max_receptors=5):

    base_eqs = '''
    dv/dt = (k*1*pF/ms/mV*(v - vr) * (v - vt) - u + I_syn_total) / C : volt
    du/dt = (1/a) * (int(v <= vb) * (b * (v - vb)**3 * cubic_current_coeff - u) - int(v > vb) * u) : amp
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
    
    return base_eqs

eqs = generate_synapse_model(max_receptors=5)
