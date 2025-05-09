from brian2 import *
import importlib
from brian2 import mV, ms, nS, Hz

def get_synapse_class(class_name):
    try:
        module = importlib.import_module('module.models.Synapse') 
        return getattr(module, class_name)
    except ModuleNotFoundError:
        raise ImportError(f"Module 'module.models.Synapse' not found.")
    except AttributeError:
        raise ImportError(f"Class '{class_name}' not found in 'module.models.Synapse'.")

def create_synapses(neuron_groups, connections, synapse_class):
    try:
        synapse_connections = []
        SynapseClass = get_synapse_class(synapse_class)
        synapse_instance = SynapseClass(neurons=neuron_groups, connections=connections)

        unit_mapping = {'nS': nS, 'ms': ms, 'mV': mV}
        param_map = {
            'AMPA': {'g': 'g_a', 'tau': 'tau_AMPA', 'E': 'E_AMPA', 'beta': 'ampa_beta'},
            'NMDA': {'g': 'g_n', 'tau': 'tau_NMDA', 'E': 'E_NMDA', 'beta': 'nmda_beta'},
            'GABA': {'g': 'g_g', 'tau': 'tau_GABA', 'E': 'E_GABA', 'beta': 'gaba_beta'},
        }

        for conn_name, conn_config in connections.items():
            pre = conn_config['pre']
            post = conn_config['post']

            if pre not in neuron_groups or post not in neuron_groups:
                print(f"Error: Neuron group {pre} or {post} not found.")
                continue

            pre_group = neuron_groups[pre]
            post_group = neuron_groups[post]

            receptor_types = conn_config['receptor_type']
            receptor_types = receptor_types if isinstance(receptor_types, list) else [receptor_types]

            for receptor_type in receptor_types:
                g0 = conn_config.get('receptor_params', {}).get(receptor_type, {}).get('g0', {})
                g0_value = g0.get('value', 0.0)
                g0_unit = unit_mapping.get(g0.get('unit', 'nS'), nS)

                on_pre_code = synapse_instance._get_on_pre(receptor_type, g0_value, pre)

                syn = Synapses(
                    pre_group, post_group,
                    model=synapse_instance.equations[receptor_type],
                    on_pre=on_pre_code
                )
                syn.connect(p=conn_config['p'])
                syn.w = conn_config.get('weight', 1.0)

                # Set receptor-specific parameters
                current_params = conn_config['receptor_params'].get(receptor_type, {})
                if receptor_type in param_map:
                    syn_var = param_map[receptor_type]
                    syn.__setattr__(syn_var['g'], g0_value * g0_unit)
                    syn.__setattr__(syn_var['tau'], current_params.get('tau_syn', {}).get('value', 1.0) * ms)
                    syn.__setattr__(syn_var['E'], current_params.get('E_rev', {}).get('value', 0.0) * mV)
                    beta_val = current_params.get('beta', {}).get('value', 1.0)
                    syn.__setattr__(syn_var['beta'], beta_val)

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
        shared_eq = 'w : 1'
        self.equations = {'AMPA': shared_eq, 'NMDA': shared_eq, 'GABA': shared_eq}

    def _get_on_pre(self, receptor_type, g0_value, pre_neuron):
        max_g = 9.5 * g0_value
        weight_term = f"0.11 * w * {g0_value} * nS" if pre_neuron.startswith("Cortex") else f"w * {g0_value} * nS"
        target = {'AMPA': 'g_a', 'NMDA': 'g_n', 'GABA': 'g_g'}[receptor_type]
        return f'''
        {target} += {weight_term}
        {target} = clip({target}, {target}, {max_g} * nS)'''

class Synapse(SynapseBase):
    def __init__(self, neurons, connections):
        super().__init__(neurons, connections)
        self.params = {'Mg2': {'value': 1.0, 'unit': '1'}}
