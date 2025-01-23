from brian2 import *
import importlib
import json
from pathlib import Path

def get_synapse_class(class_name):
    try:
        module = importlib.import_module('module.models.Synapse')
        return getattr(module, class_name)
    except (ImportError, AttributeError) as e:
        raise ImportError(f"Error loading synapse class '{class_name}': {e}")


def create_synapses(neuron_groups, connections, synapse_class):

    try:
        SynapseClass = get_synapse_class(synapse_class)
        
        synapse_instance = SynapseClass(
            neurons=neuron_groups,
            connections=connections
        )
        
        synapse_connections = []
        # print("\nAvailable neuron groups:", neuron_groups.keys())

        for conn_name, conn_config in connections.items():
            pre = conn_config['pre']
            post = conn_config['post']
            print(f"\nProcessing connection: {conn_name}")
            # print(f"Pre -> Post: {pre} -> {post}")
            
            try:
                pre_group = neuron_groups[pre]
                post_group = neuron_groups[post]
            except KeyError as e:
                print(f"Error: Could not find neuron group {e}")
                continue
            
            receptor_types = conn_config['receptor_type']
            if not isinstance(receptor_types, list):
                receptor_types = [receptor_types]
                
            for receptor_type in receptor_types:
                syn = Synapses(
                    pre_group, 
                    post_group,
                    model=synapse_instance.equations[receptor_type],
                    on_pre=synapse_instance._get_on_pre(receptor_type)
                )
                
                syn.connect(p=conn_config['p'])
                syn.w = 0.1

                params = conn_config['params']
                if isinstance(params, dict):
                    if receptor_type in params:
                        current_params = params[receptor_type]
                    else:
                        current_params = params 
                else:
                    current_params = params

                if receptor_type == 'AMPA':
                    syn.g0_a = current_params['g0']['value'] * nsiemens
                    syn.tau_AMPA = current_params['tau_syn']['value'] * ms
                    syn.E_AMPA = current_params['E_rev']['value'] * mV
                    if 'beta' in current_params:
                        syn.ampa_beta = float(current_params['beta']['value'])
                elif receptor_type == 'NMDA':
                    syn.g0_n = current_params['g0']['value'] * nsiemens
                    syn.tau_NMDA = current_params['tau_syn']['value'] * ms
                    syn.E_NMDA = current_params['E_rev']['value'] * mV
                    if 'beta' in current_params:
                        syn.nmda_beta = float(current_params['beta']['value'])
                elif receptor_type == 'GABA':
                    syn.g0_g = current_params['g0']['value'] * nsiemens
                    syn.tau_GABA = current_params['tau_syn']['value'] * ms
                    syn.E_GABA = current_params['E_rev']['value'] * mV
                    if 'beta' in current_params:
                        syn.gaba_beta = float(current_params['beta']['value'])
                
                if 'delay' in params:
                    syn.delay = params['delay']['value'] * eval(params['delay']['unit'])
                
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
                g0_a : siemens
                E_AMPA : volt
                w : 1
                ampa_beta : 1
                tau_AMPA : second
                dg_a/dt = -g_a / tau_AMPA : siemens (clock-driven)
                I_AMPA_syn = ampa_beta * w * g_a * (E_AMPA - v) : amp 
            ''',
            'NMDA': '''
                g0_n : siemens
                Mg2: 1
                w : 1
                E_NMDA : volt 
                nmda_beta: 1
                tau_NMDA : second
                dg_n/dt = -g_n / tau_NMDA : siemens (clock-driven)
                I_NMDA_syn = nmda_beta * w * g_n * (E_NMDA - v) / (1 + Mg2 * exp(-0.062 * v / mV) / 3.57) : amp 
            ''',
            'GABA': '''
                g0_g : siemens
                E_GABA : volt
                w : 1
                tau_GABA : second
                gaba_beta : 1
                dg_g/dt = -g_g / tau_GABA : siemens (clock-driven)
                I_GABA_syn = gaba_beta * w * g_g * (E_GABA - v) : amp 
            '''
        }

    def _get_on_pre(self, receptor_type):
        if receptor_type == 'AMPA':
            return '''v_post += w * mV; g_a += g0_a'''
        elif receptor_type == 'NMDA':
            return '''v_post += w * mV; g_n += g0_n'''
        elif receptor_type == 'GABA':
            return '''v_post += w * mV; g_g += g0_g'''
        else:
            raise ValueError(f"Unknown receptor type: {receptor_type}")

class Synapse(SynapseBase):
    def __init__(self, neurons, connections):
        super().__init__(neurons, connections)
        self.params = {'Mg2': {'value': 1.0, 'unit': '1'}}
