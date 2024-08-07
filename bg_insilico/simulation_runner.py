import json
import numpy as np
from brian2 import mV, pA, siemens, ms, farad, second, pF, nS, Hz
import importlib
from result import Visualization

def load_params(json_file):
    with open(json_file, 'r') as f:
        params = json.load(f)
    return params['params'], params['model']

def convert_units(params):
    converted_params = {}
    for param, info in params.items():
        value = info['value']
        unit = info['unit']
        if unit:
            # 단위를 직접 가져오기
            if unit == 'nS':
                value *= nS
            elif unit == 'mV':
                value *= mV
            elif unit == 'ms':
                value *= ms
            elif unit == 'pF':
                value *= pF
            elif unit == 'pA':
                value *= pA
            elif unit == '1/second':
                value *= Hz
            else:
                print(f"Unknown unit for {param}: {unit}")
            # 추가 단위들 필요에 따라 추가
        converted_params[param] = value
    return converted_params

def run_simulation(N, params, model_name):
    # 모델 클래스를 동적으로 로드
    module_path = f'models/{model_name}.py'

    # NeuronModel 클래스 가져오기
    spec = importlib.util.spec_from_file_location(model_name, module_path)
    model_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(model_module)
    neuron_model_class = model_module.NeuronModel

    # 파라미터 변환
    converted_params = convert_units(params)
    print("Converted parameters:", converted_params)  # 디버깅 출력
    
    # 뉴런 모델 초기화
    neuron_model = neuron_model_class(N, converted_params)
    sim = Visualization(neuron_model)
    
    # 시뮬레이션 초기 실행
    Initialize_time = 1000 * ms
    sim.run(duration=Initialize_time)
    
    # 결과 수집
    times = sim.dv_monitor.t
    v_reset = converted_params['vr']
    membrane_potential = sim.dv_monitor.v[0]
    matching_indices = np.where(membrane_potential / mV >= v_reset / mV)[0]

    if len(matching_indices) > 0:
        earliest_time_stabilized = times[matching_indices[0]] * 1000
    else:
        earliest_time_stabilized = None

    return sim
