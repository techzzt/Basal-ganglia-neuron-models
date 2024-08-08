import json
import numpy as np
from brian2 import mV, pA, siemens, ms, farad, second, pF, nS, Hz
from brian2 import Network, StateMonitor, SpikeMonitor, PopulationRateMonitor
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
   # 초기 I 값을 0으로 설정하고 이후 사용을 위해 원래 I 값을 저장
    initial_I = 0 * pA
    increase_I = converted_params['I']
    converted_params['I'] = initial_I
    
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

    # 새 모니터 설정
    dv_monitor_new = StateMonitor(neuron_model.neurons, variables='v', record=True)
    spike_monitor_new = SpikeMonitor(neuron_model.neurons)
    rate_monitor_new = PopulationRateMonitor(neuron_model.neurons)  # 10ms의 bin 크기
    current_monitor = StateMonitor(neuron_model.neurons, variables='I', record=True)

    sim.network.add(dv_monitor_new, spike_monitor_new, rate_monitor_new, current_monitor)

    # Input current
    if earliest_time_stabilized is not None:
        wait_time_after_stabilization = 100 * ms
        time_after_increase = 200 * ms
        time_after_decrease = 200 * ms
        total_simulation_time = Initialize_time + wait_time_after_stabilization + time_after_increase + time_after_decrease

        # Stabilization period
        sim.run(duration=wait_time_after_stabilization)
        
        # Increase I to the value specified in JSON
        neuron_model.neurons.I = increase_I
        sim.run(duration=time_after_increase)
        
        # Decrease I back to 0
        neuron_model.neurons.I = 0 * pA
        sim.run(duration=time_after_decrease)
    else:
        print("v did not reach v_reset, stopping simulation")
        total_simulation_time = Initialize_time

    return sim