from brian2 import *
import importlib
import json
from pathlib import Path
from brian2 import mV, ms, nS, Hz

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
        SynapseClass = get_synapse_class(synapse_class) 
        synapse_instance = SynapseClass(neurons=neuron_groups, connections=connections)

        created_synapses = {}

        unit_mapping = {'nS': nS, 'ms': ms, 'mV': mV}

        for conn_name, conn_config in connections.items():
            pre = conn_config['pre']
            post = conn_config['post']
            # print(f"\nProcessing connection: {conn_name}")

            try:
                pre_group = neuron_groups[pre]
                post_group = neuron_groups[post]
            except KeyError as e:
                print(f"Error: Could not find neuron group {e}")
                continue

            receptor_types = conn_config['receptor_type']
            receptor_types = receptor_types if isinstance(receptor_types, list) else [receptor_types]

            for receptor_type in receptor_types:
                syn_name = f"synapse_{pre}_{post}"

                if syn_name not in created_synapses:
                    g0_value = conn_config.get('receptor_params', {}).get(receptor_type, {}).get('g0', {}).get('value', 0.0)
                    # print("g0", g0_value)
                    on_pre_code = synapse_instance._get_on_pre(receptor_type, g0_value, pre)

                    syn = Synapses(
                        pre_group, 
                        post_group,
                        model=synapse_instance.equations[receptor_type],
                        on_pre=on_pre_code  
                    )
                    syn.connect(p=conn_config['p']) 
                    created_synapses[syn_name] = syn
                else:
                    syn = created_synapses[syn_name]

                syn.w = conn_config.get('weight', 1.0)
                current_params = conn_config['receptor_params'].get(receptor_type, {})

                param_map = {
                    'AMPA': {'g': 'g_a', 'tau': 'tau_AMPA', 'E': 'E_AMPA', 'beta': 'ampa_beta'},
                    'NMDA': {'g': 'g_n', 'tau': 'tau_NMDA', 'E': 'E_NMDA', 'beta': 'nmda_beta'},
                    'GABA': {'g': 'g_g', 'tau': 'tau_GABA', 'E': 'E_GABA', 'beta': 'gaba_beta'},
                }

                if receptor_type in param_map:
                    syn_var = param_map[receptor_type]
                    unit = unit_mapping.get(current_params.get('g0', {}).get('unit', 'nS'), nS)
                    syn.__setattr__(syn_var['g'], g0_value * unit) 
                    syn.__setattr__(syn_var['tau'], current_params.get('tau_syn', {}).get('value', 1.0) * ms)
                    syn.__setattr__(syn_var['E'], current_params.get('E_rev', {}).get('value', 0.0) * mV)

                    if 'beta' in current_params:
                        beta_value = current_params['beta'].get('value', 1.0)
                        syn.__setattr__(syn_var['beta'], beta_value)

                    else: 
                        syn.__setattr__(syn_var['beta'], 1.0)

                if 'delay' in current_params:
                    delay_unit = unit_mapping.get(current_params['delay'].get('unit', 'ms'), ms)
                    syn.delay = current_params['delay'].get('value', 0.0) * delay_unit
                synapse_connections.append(syn)

        return synapse_connections
        
    except Exception as e:
        print(f"Error creating synapses: {str(e)}")
        raise

class SynapseBase:
    def __init__(self, neurons, connections):
        self.neurons = neurons
        for key, value in neurons.items():
            setattr(self, key, value)
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

    def _get_on_pre(self, receptor_type, g0_value, pre_neuron):
        max_g_val = (9.5 * g0_value)
        max_g_str = f"{max_g_val} * nS"
        weight_term = f"0.11 * w * {g0_value} * nS" if pre_neuron.startswith("Cortex") else f"w * {g0_value} * nS"
        
        if receptor_type == 'AMPA':
            return f'''
            g_a += {weight_term}
            g_a = clip(g_a, g_a, {max_g_str})'''
        elif receptor_type == 'NMDA':
            return f'''
            g_n += {weight_term}
            g_n = clip(g_n, g_n, {max_g_str})'''
        elif receptor_type == 'GABA':
            return f'''
            g_g += {weight_term}
            g_g = clip(g_g, g_g, {max_g_str})'''

class Synapse(SynapseBase):
    def __init__(self, neurons, connections):
        super().__init__(neurons, connections)
        self.params = {'Mg2': {'value': 1.0, 'unit': '1'}}

