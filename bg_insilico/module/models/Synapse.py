from brian2 import *
import importlib
import json
from pathlib import Path
import numpy as np
import logging
from collections import defaultdict

logging.getLogger('brian2').setLevel(logging.ERROR)

class SynapseBase:
    def __init__(self, neurons, connections):
        self.neurons = neurons
        self.connections = connections
        self.define_equations()

    def define_equations(self):
        self.equations = {
            'AMPA': '''
                w : 1
            ''',
            'NMDA': '''
                w : 1
            ''',
            'GABA': '''
                w : 1
            '''
        }

    def _get_on_pre(self, receptor_type, g0_value):
        fac = {'AMPA': 2.2, 'NMDA': 5.0, 'GABA': 2}.get(receptor_type, 1)
        max_g_val = fac * g0_value
        max_g = f"{max_g_val}*nS"
        conductance_increase = f"w * {g0_value} * nS"

        var_map = {
            'AMPA': 'g_a', 
            'NMDA': 'g_n', 
            'GABA': 'g_g'
        }
        var = var_map.get(receptor_type)
        if var is None:
            return ''

        return f"""
        {var} += {conductance_increase}
        {var} = clip({var}, 0 * nS, {max_g})
        """

class Synapse(SynapseBase):
    def __init__(self, neurons, connections):
        super().__init__(neurons, connections)

def get_synapse_class(class_name):
    try:
        module = importlib.import_module('module.models.Synapse')
        return getattr(module, class_name)
    except ModuleNotFoundError:
        raise ImportError(f"Module 'module.models.Synapse' not found. Check if the path is correct.")
    except AttributeError:
        raise ImportError(f"Class '{class_name}' not found in 'module.models.Synapse'.")

def create_synapses(neuron_groups, connections, synapse_class_name):
    try:
        synapse_connections = []
        synapse_class = get_synapse_class(synapse_class_name) if isinstance(synapse_class_name, str) else synapse_class_name
        synapse_instance = synapse_class(neurons=neuron_groups, connections=connections)
        created_synapses_map = {}

        for conn_name, conn_config in connections.items():
            pre, post = conn_config['pre'], conn_config['post']
            pre_group, post_group = neuron_groups.get(pre), neuron_groups.get(post)

            if not pre_group or not post_group:
                continue

            receptor_types = conn_config['receptor_type']
            receptor_types = receptor_types if isinstance(receptor_types, list) else [receptor_types]

            for receptor_type in receptor_types:
                syn_key = (pre, post, receptor_type, conn_name) 

                if syn_key not in created_synapses_map:
                    if receptor_type not in synapse_instance.equations:
                        continue

                    model_eqns = synapse_instance.equations[receptor_type]
                    g0_value_for_on_pre = conn_config.get('receptor_params', {}).get(receptor_type, {})
                    g0_value_for_on_pre = g0_value_for_on_pre.get('g0', {}).get('value', 0.0)
                    
                    on_pre_code = synapse_instance._get_on_pre(receptor_type, g0_value_for_on_pre)

                    try:
                        syn = Synapses(
                            pre_group,
                            post_group,
                            model=model_eqns,
                            on_pre=on_pre_code
                        )
                        created_synapses_map[syn_key] = syn
                        synapse_connections.append(syn)

                        p_connect = conn_config.get('p', 1.0)
                        syn.connect(p=p_connect)

                        if len(syn.i) > 0:
                            weight = conn_config.get('weight', 1.0)
                            syn.w = weight
                            
                            current_params = conn_config.get('receptor_params', {}).get(receptor_type, {})
                            
                            if hasattr(syn, 'tau_' + receptor_type):
                                tau_val = current_params.get('tau_syn', {}).get('value', 160.0)
                                setattr(syn, f'tau_{receptor_type}', tau_val * ms)
                                
                            if hasattr(syn, 'E_' + receptor_type):
                                e_val = current_params.get('E_rev', {}).get('value', 0.0)
                                setattr(syn, f'E_{receptor_type}', e_val * mV)
                                
                            if hasattr(syn, receptor_type.lower() + '_beta'):
                                beta_val = current_params.get('beta', {}).get('value', 1.0)
                                setattr(syn, f'{receptor_type.lower()}_beta', beta_val)
                    except Exception as e:
                        print(f"ERROR creating {receptor_type} synapse: {str(e)}")
                else:
                    pass  

        return synapse_connections

    except Exception as e:
        print(f"Error creating synapses: {str(e)}")
        raise

def generate_neuron_specific_synapse_inputs(neuron_name, connections, already_defined=None):
    used_receptors = set()
    for conn_name, conn_config in connections.items():
        if conn_config['post'] == neuron_name:
            receptor_types = conn_config['receptor_type']
            receptor_types = receptor_types if isinstance(receptor_types, list) else [receptor_types]
            used_receptors.update(receptor_types)
    
    return generate_synapse_inputs(list(used_receptors), already_defined=already_defined)


def generate_synapse_inputs(receptors=None, already_defined=None):
    if receptors is None:
        receptors = ['AMPA', 'NMDA', 'GABA']
    if already_defined is None:
        already_defined = set()

    eqs = ''
    current_vars = []

    if 'AMPA' in receptors and 'g_a' not in already_defined:
        eqs += 'tau_AMPA : second\n'
        eqs += 'dg_a/dt = -g_a / tau_AMPA : siemens\n'
        eqs += 'ampa_beta : 1\n'
        eqs += 'E_AMPA : volt\n'
        eqs += 'I_AMPA = ampa_beta * g_a * (E_AMPA - v) : amp\n'
        current_vars.append('I_AMPA')

    if 'GABA' in receptors and 'g_g' not in already_defined:
        eqs += 'tau_GABA : second\n'
        eqs += 'dg_g/dt = -g_g / tau_GABA : siemens\n'
        eqs += 'gaba_beta : 1\n'
        eqs += 'E_GABA : volt\n'
        eqs += 'I_GABA = gaba_beta * g_g * (E_GABA - v) : amp\n'
        current_vars.append('I_GABA')

    if 'NMDA' in receptors and 'g_n' not in already_defined:
        eqs += 'tau_NMDA : second\n'
        eqs += 'dg_n/dt = -g_n / tau_NMDA : siemens\n'
        eqs += 'nmda_beta : 1\n'
        eqs += 'E_NMDA : volt\n'
        eqs += 'Mg2 : 1\n'
        eqs += 'I_NMDA = nmda_beta * g_n * (E_NMDA - v) / (1 + Mg2 * exp(-0.062 * v / mV) / 3.57) : amp\n'
        current_vars.append('I_NMDA')

    if current_vars:
        eqs += f'Isyn = {" + ".join(current_vars)} : amp\n'
    else:
        eqs += f'Isyn = 0 * amp : amp\n'

    return eqs

