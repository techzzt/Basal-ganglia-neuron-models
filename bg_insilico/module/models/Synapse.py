from brian2 import *
import importlib
import json
from pathlib import Path
import logging
logging.getLogger('brian2').setLevel(logging.ERROR)
prefs.codegen.target = 'cython'

class SynapseBase:
    def __init__(self, neurons, connections):
        self.neurons = neurons
        for name, group in neurons.items():
            setattr(self, name, group)
        self.connections = connections

    def get_on_pre(self, receptor_type, g0_value):
        max_g_dict = {
            'AMPA': 5,
            'NMDA': 2.5,
            'GABA': 0.1
        }
        max_g = max_g_dict.get(receptor_type, 1)
        weight_term = f"{g0_value} * nS"
        return f"g = clip(g + w * {weight_term}, 0*nS, {max_g}*nS)"

    def create_synapse_equation_with_idx(self, receptor_type, idx):
        """
        Create synapse equations for specific receptor type and index.
        
        Parameters:
        -----------
        receptor_type : str
            Type of receptor ('AMPA', 'NMDA', or 'GABA')
        idx : int
            Index for the synapse to allow multiple connections of same type
            
        Returns:
        --------
        str : Brian2 synapse equations
        """
        base_equation = '''
            w : 1
            tau_syn : second
            E_rev : volt
            beta : 1
            dg/dt = -g / tau_syn : siemens (clock-driven)
        '''
        
        if receptor_type == 'NMDA':
            return base_equation + f'''
            Mg2 : 1
            I_syn_{receptor_type}_{idx}_post = beta * w * g * (E_rev - v_post) / (1 + Mg2 * exp(-0.062 * v_post / mV) / 3.57) : amp (summed)
            '''
        else:
            return base_equation + f'''
            I_syn_{receptor_type}_{idx}_post = beta * w * g * (E_rev - v_post) : amp (summed)
            '''

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

def create_synapses(neuron_groups, connections, synapse_class, max_receptors=5):
    synapse_connections = []
    synapse_base = SynapseBase(neurons=neuron_groups, connections={})
    
    unit_mapping = {'nS': nS, 'ms': ms, 'mV': mV}
    
    synapse_objects = {}
    
    receptor_counters = {}
    
    for conn_name, conn_config in connections.items():
        pre, post = conn_config['pre'], conn_config['post']
        pre_group, post_group = neuron_groups.get(pre), neuron_groups.get(post)
        
        if not pre_group or not post_group:
            print(f"Error: Neuron group '{pre if not pre_group else post}' not found.")
            continue
        
        receptor_types = conn_config['receptor_type']
        receptor_types = receptor_types if isinstance(receptor_types, list) else [receptor_types]
        
        for receptor_type in receptor_types:
            if post not in receptor_counters:
                receptor_counters[post] = {}
            if receptor_type not in receptor_counters[post]:
                receptor_counters[post][receptor_type] = 0
            
            cur_idx = receptor_counters[post][receptor_type]
            if cur_idx >= max_receptors:
                print(f"Warning: Maximum number of {receptor_type} synapses ({max_receptors}) exceeded for {post}. Skipping additional connections.")
                continue
                
            idx = receptor_counters[post][receptor_type]
            receptor_counters[post][receptor_type] += 1
            
            model = synapse_base.create_synapse_equation_with_idx(receptor_type, idx)
            
            key = (pre, post, receptor_type, idx)

            current_params = conn_config['receptor_params'].get(receptor_type, {})
            g0_value = current_params.get('g0', {}).get('value', 0.0)
            
            on_pre_code = synapse_base.get_on_pre(receptor_type, g0_value)
            
            syn = Synapses(pre_group, post_group, model=model, on_pre=on_pre_code)
            synapse_objects[key] = syn
            
            p_connect = conn_config.get('p', 1.0)
            syn.connect(p=p_connect)
            
            if len(syn.i) <= 0:
                continue
                
            weight = conn_config.get('weight', 1.0)
            syn.w = weight
            
            tau_val = current_params.get('tau_syn', {}).get('value', 5.0)
            E_val = current_params.get('E_rev', {}).get('value', 0.0)
            beta_val = current_params.get('beta', {}).get('value', 1.0)
            
            syn.tau_syn = tau_val * ms
            syn.E_rev = E_val * mV
            syn.beta = beta_val
            
            if receptor_type == 'NMDA':
                syn.Mg2 = 1.0
            
            if 'delay' in current_params:
                delay_val = current_params['delay'].get('value', 0.0)
                delay_unit = unit_mapping.get(current_params['delay'].get('unit', 'ms'), ms)
                if delay_val > 0:
                    syn.delay = delay_val * delay_unit
    
    return list(synapse_objects.values())