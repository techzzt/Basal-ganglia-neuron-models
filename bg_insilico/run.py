#!/usr/bin/env python3
import json
from module.simulation.runner import run_simulation_with_inh_ext_input
from module.utils.param_loader import load_params

def main():
    # 파라미터 파일 경로
    params_file = 'config/params.json'
    
    # 파라미터 로드
    params = load_params(params_file)
    
    # 뉴런 설정 생성
    neuron_configs = params['neurons']
    
    # 시냅스 파라미터
    synapse_params = params['synapse_params']
    
    # Cortex 입력 설정
    cortex_inputs = params['cortex_inputs']
    
    synapse_class = params['synapse_class']

    # 시뮬레이션 실행
    results = run_simulation_with_inh_ext_input(
        neuron_configs=neuron_configs,
        synapse_params=synapse_params,
        synapse_class=synapse_class,
        cortex_inputs=cortex_inputs
    )
    
    print("Simulation completed successfully!")

if __name__ == "__main__":
    main()