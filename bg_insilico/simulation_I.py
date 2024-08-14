import json
import numpy as np
from brian2 import mV, pA, siemens, ms, farad, second, pF, nS, Hz, volt, ohm
from brian2 import Network, StateMonitor, SpikeMonitor, PopulationRateMonitor
import importlib
import matplotlib.pyplot as plt
from result_I import Run

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
    injection_times = []  # 입력 전류의 시점 및 지속 시간 저장할 리스트

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
        sim = Run(neuron_model)

        # 시뮬레이션 초기 실행
        Initialize_time = 1000 * ms
        neuron_model.neurons.I = 0 * pA   

        wait_time_after_stabilization = 300 * ms 
        sim.run(duration = wait_time_after_stabilization)

        # 결과 수집
        membrane_potential = sim.dv_monitor.v[0]  # Membrane potential 기록
        all_results.append(membrane_potential)

        # 전류를 저장
        current = sim.current_monitor.I[0]  # I 값을 모니터에서 저장
        all_currents.append(current)

        # 주입 전류를 입력값으로 변경
        neuron_model.neurons.I = converted_I
        time_after_increase = 200 * ms
        sim.run(duration=time_after_increase)  # 입력값으로 전류를 주입
        
        # 멤브레인 포텐셜을 다시 수집
        membrane_potential_after_increase = sim.dv_monitor.v[0]
        all_results.append(membrane_potential_after_increase)  # 새로운 결과 추가
        all_currents.append(current)  # 전류도 추가

        # 0으로 다시 설정 후 시뮬레이션
        neuron_model.neurons.I = 0 * pA
        time_after_decrease = Initialize_time - wait_time_after_stabilization - time_after_increase
        sim.run(duration=time_after_decrease)   
        membrane_potential_after_decrease = sim.dv_monitor.v[0]  # 전류가 0일 때의 멤브레인 포텐셜 기록
        all_results[-1] = np.concatenate((all_results[-1], membrane_potential_after_decrease))  # 이전 결과와 병합

        # 총 시간 수집
        total_time.append(sim.dv_monitor.t)
        
        # 입력 전류의 시점과 지속 시간 기록
        injection_times.append({
            'start': wait_time_after_stabilization / ms,
            'duration': time_after_increase / ms
        })
    total_time = np.concatenate(total_time) / ms  # ms 단위로 변환

    return all_results, all_currents, total_time, injection_times


def plot_results(all_results, all_currents, I_values, total_time, injection_times):
    num_plots = len(all_results)
    plt.figure(figsize=(15, 4 * (num_plots + 1)))  # 각 그래프에 적당한 높이 할당

    # Membrane Potential Plot
    for i, (membrane_potential, injection_time) in enumerate(zip(all_results, injection_times)):
        plt.subplot(num_plots + 1, 1, i + 1)  # 각 전류에 대해 별도의 서브플롯 생성
        plt.plot(total_time[:len(membrane_potential)], membrane_potential / mV, label=f'I = {I_values[i]} pA')
        plt.xlabel('Time (ms)')
        plt.ylabel('Membrane Potential (mV)')
        plt.title(f'Membrane Potential for I = {I_values[i]} pA')
        plt.legend()

        # 주입 전류 시점 표시
        start_time = injection_time['start']
        duration = injection_time['duration']
        plt.axvline(x=start_time, color='r', linestyle='--', label='Current Injection Start' if i == 0 else "")
        plt.axvline(x=start_time + duration, color='g', linestyle='--', label='Current Injection End' if i == 0 else "")

    # Input Current Plot
    plt.subplot(num_plots + 1, 1, num_plots + 1)

    # 전체 시간 및 전류 데이터 초기화
    total_time_current = np.arange(0, 2000, 1)  # 2000ms 동안의 시간
    total_current = []

    # 각 입력 전류에 대해 그래프 겹치기
    for I in I_values:
        input_current_pattern = []
        
        # 특정 시점에만 값이 바뀌도록 설정
        for t in range(2000):  # 0ms부터 2000ms까지
            if t < 200:
                input_current_pattern.append(0)  # 0 pA
            elif 200 <= t < 400:
                input_current_pattern.append(I)  # I pA
            else:
                input_current_pattern.append(0)  # 0 pA
        
        total_current.append(input_current_pattern)

    # Plot the input current
    for i, current in enumerate(total_current):
        plt.plot(total_time_current, current, label=f'I = {I_values[i]} pA', linestyle='--')

    plt.xlabel('Time (ms)')
    plt.ylabel('Current (pA)')
    plt.title('Input Current Patterns Over Time')
    plt.legend()
    plt.subplots_adjust(hspace=0.3)  # 수직 간격을 줄이려면 hspace 값을 조정하세요

    plt.tight_layout()  # 자동으로 서브플롯 간의 공간 조정
    plt.show()



