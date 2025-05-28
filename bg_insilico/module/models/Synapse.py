from brian2 import *
import importlib
import json
from pathlib import Path
import numpy as np

from brian2 import mV, ms, nS, Hz
import logging
from collections import defaultdict

logging.getLogger('brian2').setLevel(logging.ERROR)
prefs.codegen.target = 'cython'

class SynapseBase:
    def __init__(self, neurons, connections):
        self.neurons = neurons
        for name, group in neurons.items():
            setattr(self, name, group)
        self.connections = connections
        self.define_equations()

    def define_equations(self):
        self.equations = {
            'AMPA': '''
                w : 1
                dg_a/dt = -g_a / tau_AMPA : siemens (clock-driven)
                tau_AMPA : second
                E_AMPA : volt
                ampa_beta : 1
            ''',
            'NMDA': '''
                w : 1
                dg_n/dt = -g_n / tau_NMDA : siemens (clock-driven)
                tau_NMDA : second
                E_NMDA : volt
                nmda_beta : 1
                Mg : 1
            ''',
            'GABA': '''
                w : 1
                dg_g/dt = -g_g / tau_GABA : siemens (clock-driven)
                tau_GABA : second
                E_GABA : volt
                gaba_beta : 1
            '''
        }

    def _get_on_pre(self, receptor_type, g0_value, pre_neuron, post_neuron):
        saturation_factors = {
            'AMPA': 2,
            'NMDA': 1,
            'GABA': 0.1
        }
        fac = saturation_factors.get(receptor_type, 1)
        max_g = f"{fac * g0_value} * nS"
        
        if receptor_type == 'AMPA':
            if pre_neuron.startswith("Cortex"):
                return f"""
                g_a += w * 0.11 * {g0_value} * nS
                g_a = clip(g_a, -inf * nS, {max_g})
                I_syn_AMPA_post += ampa_beta * g_a * (E_AMPA - v_post)
                """
            else:
                return f"""
                g_a += w * {g0_value} * nS
                g_a = clip(g_a, -inf * nS, {max_g})
                I_syn_AMPA_post += ampa_beta * g_a * (E_AMPA - v_post)
                """
        elif receptor_type == 'NMDA':
            if pre_neuron.startswith("Cortex"):
                return f"""
                g_n += w * 0.11 * {g0_value} * nS
                g_n = clip(g_n, -inf * nS, {max_g})
                I_syn_NMDA_post += nmda_beta * g_n * (E_NMDA - v_post) / (1 + Mg2 * exp(-0.062 * v_post / mV) / 3.57)
                """
            else:
                return f"""
                g_n += w * {g0_value} * nS
                g_n = clip(g_n, -inf * nS, {max_g})
                I_syn_NMDA_post += nmda_beta * g_n * (E_NMDA - v_post) / (1 + Mg2 * exp(-0.062 * v_post / mV) / 3.57)
                """
        elif receptor_type == 'GABA':
            if pre_neuron.startswith("Cortex"):
                return f"""
                g_g += w * 0.11 * {g0_value} * nS
                g_g = clip(g_g, -inf * nS, {max_g})
                I_syn_GABA_post += gaba_beta * g_g * (E_GABA - v_post)
                """
            else:
                return f"""
                g_g += w * {g0_value} * nS
                g_g = clip(g_g, -inf * nS, {max_g})
                I_syn_GABA_post += gaba_beta * g_g * (E_GABA - v_post)
                """

class Synapse(SynapseBase):
    def __init__(self, neurons, connections):
        super().__init__(neurons, connections)
        self.params = {'Mg2': {'value': 1.0, 'unit': '1'}}

def get_synapse_class(class_name):
    try:
        module = importlib.import_module('module.models.Synapse') 
        return getattr(module, class_name)
    except ModuleNotFoundError:
        raise ImportError(f"Module 'module.models.Synapse' not found. Check if the path is correct.")
    except AttributeError:
        raise ImportError(f"Class '{class_name}' not found in 'module.models.Synapse'.")

def create_synapses(neuron_groups, connections, synapse_class):
    try:
        synapse_connections = []
        if isinstance(synapse_class, str):
            SynapseClass = get_synapse_class(synapse_class)
        else:
            SynapseClass = synapse_class
        synapse_instance = SynapseClass(neurons=neuron_groups, connections=connections)

        created_synapses = {}

        unit_mapping = {'nS': nS, 'ms': ms, 'mV': mV}

        for conn_name, conn_config in connections.items():
            pre, post = conn_config['pre'], conn_config['post']
            pre_group, post_group = neuron_groups.get(pre), neuron_groups.get(post)

            if not pre_group or not post_group:
                print(f"Error: Neuron group '{pre if not pre_group else post}' not found.")
                continue

            receptor_types = conn_config['receptor_type']
            receptor_types = receptor_types if isinstance(receptor_types, list) else [receptor_types]

            for receptor_type in receptor_types:
                syn_name = f"synapse_{pre}_{post}_{receptor_type}"

                if syn_name not in created_synapses:
                    g0_value = conn_config.get('receptor_params', {}).get(receptor_type, {}).get('g0', {}).get('value', 0.0)
                    on_pre_code = synapse_instance._get_on_pre(receptor_type, g0_value, pre, post)

                    syn = Synapses(
                        pre_group, 
                        post_group,
                        model=synapse_instance.equations[receptor_type],
                        on_pre=on_pre_code,
                        namespace={'Mg2': 1.0}  
                    )
                    syn.connect(p=conn_config.get('p', 1.0))
                    created_synapses[syn_name] = syn
                    synapse_connections.append(syn)
                    
                else:
                    syn = created_synapses[syn_name]
                    if syn not in synapse_connections:
                        synapse_connections.append(syn)

                syn.w = conn_config.get('weight', 1.0)
                current_params = conn_config['receptor_params'].get(receptor_type, {})

                param_map = {
                    'AMPA': {'g': 'g_a', 'tau': 'tau_AMPA', 'E': 'E_AMPA', 'beta': 'ampa_beta'},
                    'NMDA': {'g': 'g_n', 'tau': 'tau_NMDA', 'E': 'E_NMDA', 'beta': 'nmda_beta'},
                    'GABA': {'g': 'g_g', 'tau': 'tau_GABA', 'E': 'E_GABA', 'beta': 'gaba_beta'},
                }

                if receptor_type in param_map:
                    map_ = param_map[receptor_type]

                    g0_val = current_params.get('g0', {}).get('value', 0.0)
                    g0_unit = unit_mapping.get(current_params.get('g0', {}).get('unit', 'nS'), nS)
                    setattr(syn, map_['g'], g0_val * g0_unit)

                    tau_val = current_params.get('tau_syn', {}).get('value', 1.0)
                    setattr(syn, map_['tau'], tau_val * ms)
                    E_val = current_params.get('E_rev', {}).get('value', 0.0)
                    setattr(syn, map_['E'], E_val * mV)

                    beta_val = current_params.get('beta', {}).get('value', 1.0)
                    setattr(syn, map_['beta'], beta_val)

                    if 'delay' in current_params:
                        delay_val = current_params['delay'].get('value', 0.0)
                        delay_unit = unit_mapping.get(current_params['delay'].get('unit', 'ms'), ms)
                        syn.delay = delay_val * delay_unit

        return synapse_connections

    except Exception as e:
        print(f"Error creating synapses: {str(e)}")
        raise