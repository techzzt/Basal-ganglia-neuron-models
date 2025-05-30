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

        fac = {'AMPA': 5, 'NMDA': 2.5, 'GABA': 2}.get(receptor_type, 1)
        max_g_val = fac * g0_value
        max_g = f"{max_g_val}*nS"
        clip_min = "0*nS"
        conductance_increase = f"w * {g0_value} * nS"

        if receptor_type == 'AMPA':
            return f"""
            g_a += {conductance_increase}
            g_a = clip(g_a, {clip_min}, {max_g})
            """
        elif receptor_type == 'NMDA':
            return f"""
            g_n += {conductance_increase}
            g_n = clip(g_n, {clip_min}, {max_g})
            """
        elif receptor_type == 'GABA':
            return f"""
            g_g += {conductance_increase}
            g_g = clip(g_g, {clip_min}, {max_g})
            """
        else:
            return ''

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

        if isinstance(synapse_class_name, str):
            synapse_class = get_synapse_class(synapse_class_name)
        else:
            synapse_class = synapse_class_name

        synapse_instance = synapse_class(neurons=neuron_groups, connections=connections)
        unit_mapping = {'nS': nS, 'ms': ms, 'mV': mV, 'second': second, 'Hz': Hz, 'volt': volt, 'siemens': siemens, 'farad': farad, 'amp': amp, 'pA': pA, 'pF': pF}
        created_synapses_map = {}

        # First pass: create synapse objects and connections
        for conn_name, conn_config in connections.items():
            pre, post = conn_config['pre'], conn_config['post']
            pre_group, post_group = neuron_groups.get(pre), neuron_groups.get(post)

            if not pre_group or not post_group:
                print(f"Error: Neuron group '{pre if not pre_group else post}' not found. Skipping connection '{conn_name}'.")
                continue

            receptor_types = conn_config['receptor_type']
            receptor_types = receptor_types if isinstance(receptor_types, list) else [receptor_types]

            for receptor_type in receptor_types:
                syn_key = (pre, post, receptor_type)

                if syn_key not in created_synapses_map:
                    if receptor_type not in synapse_instance.equations:
                        print(f"Error: Equations for receptor type '{receptor_type}' not defined in SynapseBase. Skipping connection '{conn_name}'.")
                        continue

                    model_eqns = synapse_instance.equations[receptor_type]
                    g0_value_for_on_pre = conn_config.get('receptor_params', {}).get(receptor_type, {}).get('g0', {}).get('value', 0.0)
                    on_pre_code = synapse_instance._get_on_pre(receptor_type, g0_value_for_on_pre)
                    syn_object_name = f'synapses_{pre}_to_{post}_{receptor_type}'

                    syn = Synapses(
                        pre_group,
                        post_group,
                        model=model_eqns,
                        on_pre=on_pre_code,
                        name=syn_object_name
                    )
                    created_synapses_map[syn_key] = syn
                    synapse_connections.append(syn)

                syn = created_synapses_map[syn_key]
                p_connect = conn_config.get('p', 1.0)
                syn.connect(p=p_connect)

                if len(syn.i) == 0:
                    print(f"Warning: No connections made for {conn_name} ({receptor_type}) with probability {p_connect}. Skipping parameter assignment.")
                    continue

                weight = conn_config.get('weight', 1.0)
                syn.w = weight

                if 'delay' in conn_config.get('receptor_params', {}).get(receptor_type, {}):
                    delay_params = conn_config['receptor_params'][receptor_type]['delay']
                    delay_val = delay_params.get('value', 0.0)
                    delay_unit_str = delay_params.get('unit', 'ms')
                    delay_unit = unit_mapping.get(delay_unit_str, ms)
                    if delay_val >= 0:
                        syn.delay = delay_val * delay_unit

        # Second pass: set neuron group parameters with validation
        for neuron_name, neuron_group in neuron_groups.items():
            # Set default values first to prevent division by zero
            if hasattr(neuron_group, 'tau_AMPA'):
                neuron_group.tau_AMPA = 12.0 * ms  # Default AMPA tau
            if hasattr(neuron_group, 'tau_NMDA'):
                neuron_group.tau_NMDA = 160.0 * ms  # Default NMDA tau
            if hasattr(neuron_group, 'tau_GABA'):
                neuron_group.tau_GABA = 8.0 * ms  # Default GABA tau
            if hasattr(neuron_group, 'E_AMPA'):
                neuron_group.E_AMPA = 0.0 * mV
            if hasattr(neuron_group, 'E_NMDA'):
                neuron_group.E_NMDA = 0.0 * mV
            if hasattr(neuron_group, 'E_GABA'):
                neuron_group.E_GABA = -74.0 * mV
            if hasattr(neuron_group, 'ampa_beta'):
                neuron_group.ampa_beta = 1.0
            if hasattr(neuron_group, 'nmda_beta'):
                neuron_group.nmda_beta = 1.0
            if hasattr(neuron_group, 'gaba_beta'):
                neuron_group.gaba_beta = 1.0
            if hasattr(neuron_group, 'Mg2'):
                neuron_group.Mg2 = 1.0
                
            # Override with connection-specific parameters if available
            for conn_name, conn_config in connections.items():
                if conn_config['post'] == neuron_name:
                    receptor_types = conn_config['receptor_type']
                    receptor_types = receptor_types if isinstance(receptor_types, list) else [receptor_types]
                    
                    for receptor_type in receptor_types:
                        current_params = conn_config.get('receptor_params', {}).get(receptor_type, {})
                        
                        if receptor_type == 'AMPA':
                            if hasattr(neuron_group, 'tau_AMPA'):
                                tau_val = current_params.get('tau_syn', {}).get('value', 12.0)
                                # Ensure tau is never zero or negative
                                tau_val = max(tau_val, 0.1)  # Minimum 0.1 ms
                                neuron_group.tau_AMPA = tau_val * ms
                            if hasattr(neuron_group, 'E_AMPA'):
                                E_val = current_params.get('E_rev', {}).get('value', 0.0)
                                neuron_group.E_AMPA = E_val * mV
                            if hasattr(neuron_group, 'ampa_beta'):
                                beta_val = current_params.get('beta', {}).get('value', 1.0)
                                neuron_group.ampa_beta = beta_val
                                
                        elif receptor_type == 'NMDA':
                            if hasattr(neuron_group, 'tau_NMDA'):
                                tau_val = current_params.get('tau_syn', {}).get('value', 160.0)
                                # Ensure tau is never zero or negative
                                tau_val = max(tau_val, 0.1)  # Minimum 0.1 ms
                                neuron_group.tau_NMDA = tau_val * ms
                            if hasattr(neuron_group, 'E_NMDA'):
                                E_val = current_params.get('E_rev', {}).get('value', 0.0)
                                neuron_group.E_NMDA = E_val * mV
                            if hasattr(neuron_group, 'nmda_beta'):
                                beta_val = current_params.get('beta', {}).get('value', 1.0)
                                neuron_group.nmda_beta = beta_val
                            if hasattr(neuron_group, 'Mg2'):
                                neuron_group.Mg2 = 1.0
                                
                        elif receptor_type == 'GABA':
                            if hasattr(neuron_group, 'tau_GABA'):
                                tau_val = current_params.get('tau_syn', {}).get('value', 8.0)
                                # Ensure tau is never zero or negative
                                tau_val = max(tau_val, 0.1)  # Minimum 0.1 ms
                                neuron_group.tau_GABA = tau_val * ms
                            if hasattr(neuron_group, 'E_GABA'):
                                E_val = current_params.get('E_rev', {}).get('value', -74.0)
                                neuron_group.E_GABA = E_val * mV
                            if hasattr(neuron_group, 'gaba_beta'):
                                beta_val = current_params.get('beta', {}).get('value', 1.0)
                                neuron_group.gaba_beta = beta_val

        return synapse_connections

    except Exception as e:
        print(f"Error creating synapses: {str(e)}")
        raise

def generate_synapse_inputs(receptors=None):
    if receptors is None:
        receptors = ['AMPA', 'NMDA', 'GABA']

    eqs = ''
    if 'AMPA' in receptors:
        eqs += 'I_AMPA = ampa_beta * g_a * (E_AMPA - v) : amp\n'
    if 'GABA' in receptors:
        eqs += 'I_GABA = gaba_beta * g_g * (E_GABA - v) : amp\n'
    if 'NMDA' in receptors:
        eqs += 'I_NMDA = nmda_beta * g_n * (E_NMDA - v) / (1 + Mg2 * exp(-0.062 * v / mV) / 3.57) : amp\n'

    total_terms = []
    if 'AMPA' in receptors: total_terms.append('I_AMPA')
    if 'NMDA' in receptors: total_terms.append('I_NMDA')
    if 'GABA' in receptors: total_terms.append('I_GABA')

    if total_terms:
        eqs += f'Isyn = {" + ".join(total_terms)} : amp\n'
    else:
        eqs += f'Isyn = 0 * amp : amp\n'

    return eqs
