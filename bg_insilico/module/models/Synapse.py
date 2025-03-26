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
        synapse_connections = []  
        SynapseClass = get_synapse_class(synapse_class) 
        
        synapse_instance = SynapseClass(neurons=neuron_groups, connections=connections)

        created_synapses = {} 

        for conn_name, conn_config in connections.items():
            pre = conn_config['pre']
            post = conn_config['post']
            print(f"\nProcessing connection: {conn_name}")

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
                syn_name = f"synapse_{pre}_{post}_{receptor_type}"

                if syn_name not in created_synapses:
                    
                    syn = Synapses(
                        pre_group, 
                        post_group,
                        model=synapse_instance.equations[receptor_type],
                        on_pre=synapse_instance._get_on_pre(receptor_type)
                    )
                    syn.connect(p=conn_config['p']) 
                    created_synapses[syn_name] = syn
                else:
                    syn = created_synapses[syn_name]
                
                syn.w = conn_config.get('weight', 1.0)
                params = conn_config['receptor_params']

                current_params = params.get(receptor_type, {})

                if receptor_type == 'AMPA':
                    syn.g_a = current_params['g0']['value'] * eval(current_params['g0']['unit'])   
                    syn.tau_AMPA = current_params['tau_syn']['value'] * ms
                    syn.E_AMPA = current_params['E_rev']['value'] * mV
                    # print("ampa", syn.g_a)
                    if 'beta' in current_params:
                        syn.ampa_beta = float(current_params['beta']['value'])
               
                if receptor_type == 'NMDA':
                    syn.g_n = current_params['g0']['value'] * eval(current_params['g0']['unit'])   
                    syn.tau_NMDA = current_params['tau_syn']['value'] * ms
                    syn.E_NMDA = current_params['E_rev']['value'] * mV
                    if 'beta' in current_params:
                        syn.nmda_beta = float(current_params['beta']['value'])
                
                if receptor_type == 'GABA':
                    syn.g_g = current_params['g0']['value'] * eval(current_params['g0']['unit'])   
                    syn.tau_GABA = current_params['tau_syn']['value'] * ms
                    syn.E_GABA = current_params['E_rev']['value'] * mV
                    if 'beta' in current_params:
                        syn.gaba_beta = float(current_params['beta']['value'])
                
                if 'delay' in current_params:
                    syn.delay = current_params['delay']['value'] * eval(current_params['delay']['unit'])             

                synapse_connections.append(syn)
                print(f"Created synapse: {syn_name}")

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

    def _get_on_pre(self, receptor_type):
        if receptor_type == 'AMPA':
            return '''g_a += w * nS'''
        if receptor_type == 'NMDA':
            return '''g_n += w * nS'''
        if receptor_type == 'GABA':
            return '''g_g += w * nS'''

class Synapse(SynapseBase):
    def __init__(self, neurons, connections):
        super().__init__(neurons, connections)
        self.params = {'Mg2': {'value': 1.0, 'unit': '1'}}

