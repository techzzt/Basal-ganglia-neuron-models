from brian2 import *
from module.models.neuron_models import create_neurons
from module.models.synapse_models import create_synapses
from module.utils.data_handler import plot_raster

def run_simulation_with_inh_ext_input(neuron_configs, synapse_params, synapse_class, cortex_inputs, simulation_duration=1000*ms):

    try:
        neuron_groups = create_neurons(neuron_configs)
        print(f"생성된 뉴런 그룹: {neuron_groups.keys()}") 

        synapse_connections = create_synapses(neuron_groups, synapse_params, synapse_class)
        
        # 네트워크 생성
        net = Network()
        net.add(neuron_groups.values())
        net.add(synapse_connections)
        
        # 모니터 설정
        spike_monitors = {}
        voltage_monitors = {}
        for name, group in neuron_groups.items():
            
            # 스파이크 모니터 추가
            spike_mon = SpikeMonitor(group)
            spike_monitors[name] = spike_mon
            net.add(spike_mon)
            
            # 전압 모니터 추가 (변수 존재 여부 확인)
            if 'v' in group.variables:
                voltage_mon = StateMonitor(group, 'v', record=True)
                voltage_monitors[name] = voltage_mon
                net.add(voltage_mon)
                print(f"{name}: 전압 모니터 추가됨")
            else:
                print(f"경고: {name} 그룹에 'v' 변수가 없습니다")
                # 사용 가능한 변수 출력
                print(f"사용 가능한 변수들: {list(group.variables.keys())}")
        
        net.run(1000 * ms)
        plot_raster(spike_monitors)  
        
        results = {
            'spike_monitors': spike_monitors,
            'voltage_monitors': voltage_monitors,
            'neuron_groups': neuron_groups,
            'synapse_connections': synapse_connections
        }
        
        return results
        
    except Exception as e:
        print(f"시뮬레이션 실행 중 오류 발생: {str(e)}")
        raise
        
    except Exception as e:
        print(f"시뮬레이션 실행 중 오류 발생: {str(e)}")
        raise