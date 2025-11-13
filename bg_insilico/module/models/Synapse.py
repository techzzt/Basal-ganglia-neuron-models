# -*- coding: utf-8 -*-
# Copyright (c) 2025. All rights reserved.
# Author: keun (Jieun Kim)

from brian2 import *
import importlib
import logging
import numpy as np

logging.getLogger('brian2').setLevel(logging.ERROR)

# Base synapse class
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

    def _get_on_pre(self, receptor_type, g0_value, tau_value, weight, conn_name=None):
        g_increase = f"w * {g0_value} * nS"
        var_map = {'AMPA': 'g_a', 'NMDA': 'g_n', 'GABA': 'g_g'}
        var = var_map.get(receptor_type)

        is_ext_input = False
        if conn_name:
            name_lower = conn_name.lower()
            is_ext_input = name_lower.startswith('ext') or name_lower.startswith('cortex')
        if is_ext_input:
            return f"\n{var} += {g_increase}\n"
    
        max_g_limits = {'AMPA': 4.6, 'NMDA': 2.3, 'GABA': 5.5}
        hard_limit = max_g_limits.get(receptor_type, float('inf'))

        if tau_value is not None and tau_value > 0:
            calculated_limit = tau_value * g0_value
            
            min_saturation_limits = {
                'AMPA': 1.0,   
                'NMDA': 0.5,     
                'GABA': 1   
            }

            min_limit = min_saturation_limits.get(receptor_type, 0.5)
            calculated_limit = max(calculated_limit, min_limit)
            
            safety_multiplier = 1.5
            max_g_value = min(calculated_limit, hard_limit * safety_multiplier)
        else:
            calculated_limit = float('inf')
            max_g_value = hard_limit

        if max_g_value == float('inf'):
            return f"\n{var} += {g_increase}\n"

        return f'''
        {var} = clip({var} + {g_increase}, 0 * nS, {max_g_value} * nS)
        '''

# Main synapse class
class Synapse(SynapseBase):
    def __init__(self, neurons, connections):
        super().__init__(neurons, connections)

# Get synapse class by name
def get_synapse_class(class_name):
    try:
        module = importlib.import_module('module.models.Synapse')
        return getattr(module, class_name)
    except ModuleNotFoundError:
        raise ImportError(f"Module 'module.models.Synapse' not found. Check if the path is correct.")
    except AttributeError:
        raise ImportError(f"Class '{class_name}' not found in 'module.models.Synapse'.")

# Create synaptic connections
def create_synapses(neuron_groups, connections, synapse_class_name, dop_cfg: dict = None):
    try:
        synapse_connections = []
        synapse_class = get_synapse_class(synapse_class_name) if isinstance(synapse_class_name, str) else synapse_class_name
        synapse_instance = synapse_class(neurons=neuron_groups, connections=connections)

        created_synapses_map = {}

        def _alpha_scale(cfg: dict):
            try:
                if not cfg or not cfg.get('enabled', False):
                    return 0.0
                a0 = float(cfg.get('alpha0', 0.8))
                ad = float(cfg.get('alpha_dop', 1.0))
                phi = ad - a0
                if not np.isfinite(phi):
                    return 0.0
                return float(phi)
            except Exception:
                return 0.0

        for conn_name, conn_config in connections.items():
            pre, post = conn_config['pre'], conn_config['post']
            pre_group, post_group = neuron_groups.get(pre), neuron_groups.get(post)
            if not pre_group or not post_group:
                continue

            receptor_types = conn_config['receptor_type']
            receptor_types = receptor_types if isinstance(receptor_types, list) else [receptor_types]
            
            weight_from_config = conn_config.get('weight', 1.0)
            alpha_value = _alpha_scale(dop_cfg)

            for receptor_type in receptor_types:
                syn_key = (pre, post, receptor_type, conn_name)

                if syn_key in created_synapses_map:
                    continue

                if receptor_type not in synapse_instance.equations:
                    continue

                model_eqns = synapse_instance.equations[receptor_type]
                current_params = conn_config.get('receptor_params', {}).get(receptor_type, {})
                g0_value_for_on_pre = current_params.get('g0', {}).get('value', 0.0)
                tau_value_for_on_pre = current_params.get('tau_syn', {}).get('value', None)

                on_pre_code = synapse_instance._get_on_pre(
                    receptor_type, 
                    g0_value_for_on_pre, 
                    tau_value_for_on_pre,
                    weight=weight_from_config,
                    conn_name=conn_name
                )

                tau_syn_dict = current_params.get('tau_syn', {})
                tau_val = tau_syn_dict.get('value', None) if tau_syn_dict else None
                erev_dict = current_params.get('E_rev', {})
                erev_val = erev_dict.get('value', None) if erev_dict else None
                beta_dict = current_params.get('beta', {})
                beta_val = beta_dict.get('value', None) if beta_dict else None
                
                try:
                    if receptor_type == 'AMPA':
                        if tau_val is not None and hasattr(post_group, 'tau_AMPA'):
                            post_group.tau_AMPA = tau_val * ms
                        if erev_val is not None and hasattr(post_group, 'E_AMPA'):
                            post_group.E_AMPA = erev_val * mV
                        if beta_val is not None and hasattr(post_group, 'ampa_beta'):
                            post_group.ampa_beta = float(beta_val)
                    elif receptor_type == 'GABA':
                        if tau_val is not None and hasattr(post_group, 'tau_GABA'):
                            post_group.tau_GABA = tau_val * ms
                        if erev_val is not None and hasattr(post_group, 'E_GABA'):
                            post_group.E_GABA = erev_val * mV
                        if beta_val is not None and hasattr(post_group, 'gaba_beta'):
                            post_group.gaba_beta = float(beta_val)
                    elif receptor_type == 'NMDA':
                        if tau_val is not None and hasattr(post_group, 'tau_NMDA'):
                            post_group.tau_NMDA = tau_val * ms
                        if erev_val is not None and hasattr(post_group, 'E_NMDA'):
                            post_group.E_NMDA = erev_val * mV
                        if beta_val is not None and hasattr(post_group, 'nmda_beta'):
                            post_group.nmda_beta = float(beta_val)
                except Exception:
                    pass
                
                try:

                    syn = Synapses(
                        pre_group,
                        post_group,
                        model=model_eqns,
                        on_pre=on_pre_code
                    )
                    created_synapses_map[syn_key] = syn
                    synapse_connections.append(syn)

                    is_external_pre = pre.startswith('Cortex_') or pre.startswith('Ext_')
                    if is_external_pre:
                        N_pre = int(pre_group.N)
                        N_post = int(post_group.N)
                        if N_pre == N_post:
                            syn.connect(j='i')
                        elif N_pre % N_post == 0 and N_post > 0:
                            syn.connect(i=np.arange(N_pre), j=(np.arange(N_pre) % N_post))
                        else:
                            min_n = min(N_pre, N_post)
                            syn.connect(i=np.arange(min_n), j=np.arange(min_n))
                    else:
                        p = conn_config.get('p', 1.0)
                        N_pre = pre_group.N

                        try:
                            N_beta_val = current_params.get('N_beta', {}).get('value', 1.0)
                            if N_beta_val is not None:
                                N_beta_val = float(N_beta_val)
                            else:
                                N_beta_val = 1.0
                        except Exception:
                            N_beta_val = 1.0
                        N_beta_scale = 1.0 + N_beta_val * float(alpha_value)
                        
                        if not np.isfinite(N_beta_scale):
                            N_beta_scale = 1.0
                        
                        fan_in = p * N_pre * N_beta_scale
                        effective_p = fan_in / N_pre
                        effective_p = min(effective_p, 1.0)
                        syn.connect(p=effective_p)

                    try:
                        beta_val_for_weight = float(beta_val) if beta_val is not None else 0.0
                        scale = 1.0 + beta_val_for_weight * float(alpha_value)
                        if not np.isfinite(scale):
                            scale = 1.0
                    except Exception:
                        scale = 1.0
                    
                    syn.w = weight_from_config * scale

                    try:
                        delay_dict = current_params.get('delay', {})
                        delay_val_ms = delay_dict.get('value', None) if delay_dict else None
                        if delay_val_ms is not None:
                            syn.delay = delay_val_ms * ms
                    except Exception:
                        pass

                except Exception:
                    pass

        return synapse_connections, created_synapses_map

    except Exception as e:
        raise

# Generate synapse inputs for specific neuron
def generate_neuron_specific_synapse_inputs(neuron_name, connections, already_defined=None):
    used_receptors = set()
    for conn_name, conn_config in connections.items():
        if conn_config['post'] == neuron_name:
            receptor_types = conn_config['receptor_type']
            receptor_types = receptor_types if isinstance(receptor_types, list) else [receptor_types]
            used_receptors.update(receptor_types)

    return generate_synapse_inputs(list(used_receptors), already_defined=already_defined)

# Generate synapse equations
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
        eqs += 'I_NMDA = nmda_beta * g_n * (E_NMDA - v) / (1 + (Mg2/3.57) * exp(-0.062 * v / mV)) : amp\n'
        current_vars.append('I_NMDA')

    if current_vars:
        eqs += f'Isyn = {" + ".join(current_vars)} : amp\n'
    else:
        eqs += f'Isyn = 0 * amp : amp\n'

    return eqs