import json
import numpy as np
from brian2 import mV, pA, siemens, ms, farad, second
import importlib
from result import Visualization

def load_params(json_file):
    with open(json_file, 'r') as f:
        params = json.load(f)
    return params['params'], params['model']

def convert_units(params):
    # 단위를 포함한 값으로 변환
    converted_params = {}
    for key, value in params.items():
        param_value = value['value']
        param_unit = value['unit']
        
        if param_unit == 'siemens':
            converted_params[key] = param_value * siemens
        elif param_unit == 'mV':
            converted_params[key] = param_value * mV
        elif param_unit == 'ms':
            converted_params[key] = param_value * ms
        elif param_unit == 'pF':
            converted_params[key] = param_value * farad  # pF는 farad로 변환
        elif param_unit == 'pA':
            converted_params[key] = param_value * pA
        elif param_unit == '1/second':
            converted_params[key] = param_value / second
        elif param_unit == '':
            converted_params[key] = param_value  # 단위가 없으면 그냥 값으로
        else:
            raise ValueError(f"Unknown unit: {param_unit}")

    return converted_params

def run_simulation(N, params, v_reset, model_name):
    # 모델 클래스를 동적으로 로드
    model_module = importlib.import_module(f'models.{model_name}')  # models 디렉토리에서 모델 로드
    neuron_model_class = getattr(model_module, 'NeuronModel')
    
    # 파라미터 변환
    converted_params = convert_units(params)  # 단위 변환 추가

    # 뉴런 모델 초기화
    neuron_model = neuron_model_class(N, converted_params)
    sim = Visualization(neuron_model)
    
    # 시뮬레이션 초기 실행
    Initialize_time = 1000 * ms
    sim.run(duration=Initialize_time)
    
    # 결과 수집
    times = sim.dv_monitor.t
    membrane_potential = sim.dv_monitor.v[0]
    matching_indices = np.where(membrane_potential / mV >= v_reset / mV)[0]

    if len(matching_indices) > 0:
        earliest_time_stabilized = times[matching_indices[0]] * 1000
    else:
        earliest_time_stabilized = None

    return sim, earliest_time_stabilized
