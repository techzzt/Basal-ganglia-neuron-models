from brian2 import *
import importlib
import json
from pathlib import Path
import logging
logging.getLogger('brian2').setLevel(logging.ERROR)

class SynapseBase:
    def __init__(self, neurons, connections):
        self.neurons = neurons
        for name, group in neurons.items():
            setattr(self, name, group)
        self.connections = connections
        self.equations = {}
        self.equations['AMPA'] = self.define_ampa_equations()
        self.equations['NMDA'] = self.define_nmda_equations()
        self.equations['GABA'] = self.define_gaba_equations()

    def define_ampa_equations(self):
        return '''
        w : 1
        tau_syn : second
        E_rev : volt
        beta : 1
        dg/dt = -g / tau_syn : siemens (clock-driven)
        I_AMPA_post = beta * w * g * (E_rev - v_post) : amp (summed)
        '''

    def define_nmda_equations(self):
        return '''
        w : 1
        tau_syn : second
        E_rev : volt
        beta : 1
        Mg2 : 1
        dg/dt = -g / tau_syn : siemens (clock-driven)
        I_NMDA_post = beta * w * g * (E_rev - v_post) / (1 + Mg2 * exp(-0.062 * v_post / mV) / 3.57) : amp (summed)
        '''

    def define_gaba_equations(self):
        return '''
        w : 1
        tau_syn : second
        E_rev : volt
        beta : 1
        dg/dt = -g / tau_syn : siemens (clock-driven)
        I_GABA_post = beta * w * g * (E_rev - v_post) : amp (summed)
        '''

    def get_on_pre(self, receptor_type, g0_value):
        max_g_dict = {
            'AMPA': 700,
            'NMDA': 350,
            'GABA': 10
        }
        max_g = max_g_dict.get(receptor_type, 1)
        weight_term = f"{g0_value} * nS"
        return f"g = clip(g + w * {weight_term}, 0*nS, {max_g}*nS)"


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
    synapse_connections = []
    synapse_base = SynapseBase(neurons=neuron_groups, connections={})
    
    unit_mapping = {'nS': nS, 'ms': ms, 'mV': mV}
    
    # 각 뉴런 그룹과 수용체 타입별로 시냅스 객체를 하나만 생성하기 위한 딕셔너리
    synapse_objects = {}
    
    for conn_name, conn_config in connections.items():
        pre, post = conn_config['pre'], conn_config['post']
        pre_group, post_group = neuron_groups.get(pre), neuron_groups.get(post)
        
        if not pre_group or not post_group:
            print(f"Error: Neuron group '{pre if not pre_group else post}' not found.")
            continue
        
        receptor_types = conn_config['receptor_type']
        receptor_types = receptor_types if isinstance(receptor_types, list) else [receptor_types]
        
        for receptor_type in receptor_types:
            # 각 (post, receptor_type) 조합에 대해 하나의 시냅스 객체만 생성
            key = (post, receptor_type)
            
            if key not in synapse_objects:
                # 해당 수용체 타입의 방정식 가져오기
                model = synapse_base.equations[receptor_type]
                
                # 수용체 타입별 파라미터
                current_params = conn_config['receptor_params'].get(receptor_type, {})
                g0_value = current_params.get('g0', {}).get('value', 0.0)
                
                # on_pre 코드 생성
                on_pre_code = synapse_base.get_on_pre(receptor_type, g0_value)
                
                # 시냅스 객체 생성
                syn = Synapses(pre_group, post_group, model=model, on_pre=on_pre_code)
                synapse_objects[key] = syn
            else:
                # 이미 생성된 시냅스 객체 사용
                syn = synapse_objects[key]
            
            # 연결 설정
            p_connect = conn_config.get('p', 1.0)
            existing_synapses = len(syn.i) if hasattr(syn, 'i') else 0
            
            # 연결 생성
            syn.connect(p=p_connect)
            
            # 새로 추가된 시냅스에 대한 인덱스
            new_synapses = len(syn.i) - existing_synapses
            if new_synapses <= 0:
                continue
                
            new_indices = slice(existing_synapses, len(syn.i))
            
            # 가중치 설정
            weight = conn_config.get('weight', 1.0)
            syn.w[new_indices] = weight
            
            # 수용체 타입별 파라미터 설정
            current_params = conn_config['receptor_params'].get(receptor_type, {})
            tau_val = current_params.get('tau_syn', {}).get('value', 5.0)
            E_val = current_params.get('E_rev', {}).get('value', 0.0)
            beta_val = current_params.get('beta', {}).get('value', 1.0)
            
            syn.tau_syn = tau_val * ms
            syn.E_rev = E_val * mV
            syn.beta = beta_val
            
            if receptor_type == 'NMDA':
                syn.Mg2 = 1.0
            
            # 지연 설정
            if 'delay' in current_params:
                delay_val = current_params['delay'].get('value', 0.0)
                delay_unit = unit_mapping.get(current_params['delay'].get('unit', 'ms'), ms)
                if delay_val > 0:
                    syn.delay = delay_val * delay_unit
    
    # 모든 시냅스 객체를 리스트에 추가
    for syn in synapse_objects.values():
        synapse_connections.append(syn)
    
    return synapse_connections