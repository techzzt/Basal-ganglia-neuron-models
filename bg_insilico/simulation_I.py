import json
import numpy as np
from brian2 import mV, pA, siemens, ms, farad, second, pF, nS, Hz, volt, ohm
from brian2 import Network, StateMonitor, SpikeMonitor, PopulationRateMonitor
import importlib
import matplotlib.pyplot as plt
from result_I import Visualization

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
            elif unit == 'Hz':
                value *= Hz 
            elif unit == '1/second':
                value *= Hz
            elif unit == 'volt/second':  
                value *= volt / second
            elif unit == 'Ohm':  
                value *= ohm
            else:
                print(f"Unknown unit for {param}: {unit}")
        converted_params[param] = value
    return converted_params


def run_simulation(N, params, model_name, I_values):
    module_path = f'models/{model_name}.py'
    spec = importlib.util.spec_from_file_location(model_name, module_path)
    model_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(model_module)
    neuron_model_class = model_module.NeuronModel

    converted_params = convert_units(params)
    
    # 결과를 저장할 리스트 초기화
    all_results = []
    all_currents = []  # I 값을 저장할 리스트
    total_time = []  # 총 시간을 저장할 리스트

    for I in I_values:
        # 주입 전류 값을 설정
        if 'I' in params:
            I_info = params['I']
            I_unit = I_info['unit']
            
            if I_unit == 'pA':
                converted_I = I * pA  # 현재 I 값을 pA 단위로 변환
            elif I_unit == 'volt/second':
                converted_I = I * (volt / second) * 1e12  # Convert to pA
            else:
                print(f"Unknown unit for I: {I_unit}")
                converted_I = 0 * pA  # Default to 0 if unknown
        else:
            converted_I = 0 * pA  # Default to 0 if I is not defined

        converted_params['I'] = converted_I

        # 뉴런 모델 초기화
        neuron_model = neuron_model_class(N, converted_params)
        sim = Visualization(neuron_model)

        # 시뮬레이션 초기 실행
        Initialize_time = 1000 * ms
        sim.run(duration=Initialize_time)

        # 결과 수집
        membrane_potential = sim.dv_monitor.v[0]  # Membrane potential 기록
        all_results.append(membrane_potential)

        # 전류를 저장
        current = sim.current_monitor.I[0]  # I 값을 모니터에서 저장
        all_currents.append(current)

        # 새 모니터 설정
        dv_monitor_new = StateMonitor(neuron_model.neurons, variables='v', record=True)
        spike_monitor_new = SpikeMonitor(neuron_model.neurons)
        rate_monitor_new = PopulationRateMonitor(neuron_model.neurons)
        current_monitor_new = StateMonitor(neuron_model.neurons, variables='I', record=True)

        sim.network.add(dv_monitor_new, spike_monitor_new, rate_monitor_new, current_monitor_new)

        # v_reset 기준으로 멤브레인 포텐셜이 안정화되었는지 확인
        v_reset = converted_params['vr']
        matching_indices = np.where(membrane_potential / mV >= v_reset / mV)[0]

        # Stabilization period and further simulation
        if len(matching_indices) > 0:
            wait_time_after_stabilization = 100 * ms
            time_after_increase = 200 * ms
            time_after_decrease = 200 * ms

            sim.run(duration=wait_time_after_stabilization)
            neuron_model.neurons.I = converted_I
            sim.run(duration=time_after_increase)
            neuron_model.neurons.I = 0 * pA
            sim.run(duration=time_after_decrease)

            # 총 시간 수집
            total_time.append(sim.dv_monitor.t)
    total_time = np.concatenate(total_time) / ms  # ms 단위로 변환

    return all_results, all_currents, total_time


def plot_results(all_results, all_currents, I_values, total_time):
    plt.figure(figsize=(15, 10))

    # Membrane Potential Plot
    plt.subplot(2, 1, 1)
    for i, membrane_potential in enumerate(all_results):
        # x축을 total_time으로 설정하고, y축을 membrane_potential로 설정
        plt.plot(total_time[:len(membrane_potential)], membrane_potential / mV, label=f'I = {I_values[i]} pA')
    plt.xlabel('Time (ms)')
    plt.ylabel('Membrane Potential (mV)')
    plt.title('Membrane Potential over Time')
    plt.legend()

    # Input Current Plot
    plt.subplot(2, 1, 2)
    for i, current in enumerate(all_currents):
        plt.plot(total_time[:len(current)], current / pA, label=f'I = {I_values[i]} pA')
    plt.xlabel('Time (ms)')
    plt.ylabel('Input Current (pA)')
    plt.title('Input Current over Time')
    plt.legend()

    plt.tight_layout()
    plt.show()

