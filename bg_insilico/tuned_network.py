#!/usr/bin/env python3
import json
import time
import matplotlib.pyplot as plt
import numpy as np
from module.simulation.runner import run_simulation_with_inh_ext_input
from module.utils.param_loader import load_params
from brian2 import ms 

def plot_raster(results, plot_order, duration_ms=1500):
    """Create raster plots for all neuron types"""
    fig, axes = plt.subplots(len(plot_order), 1, figsize=(14, 2.5*len(plot_order)), sharex=True)
    if len(plot_order) == 1:
        axes = [axes]
    
    colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown']
    
    for i, neuron_type in enumerate(plot_order):
        if hasattr(results, 'spike_trains') and neuron_type in results.spike_trains:
            spike_times = results.spike_trains[neuron_type].t / ms
            spike_indices = results.spike_trains[neuron_type].i
            
            if len(spike_times) > 0:
                axes[i].scatter(spike_times, spike_indices, s=0.8, alpha=0.7, color=colors[i % len(colors)])
                axes[i].set_ylabel(f'{neuron_type}\nNeuron #')
                axes[i].set_xlim(0, duration_ms)
                
                n_neurons = len(np.unique(spike_indices)) if len(spike_indices) > 0 else 0
                total_spikes = len(spike_times)
                if n_neurons > 0:
                    rate = total_spikes / n_neurons / (duration_ms / 1000)
                    axes[i].text(0.02, 0.95, f'{rate:.2f} Hz', transform=axes[i].transAxes, 
                               bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
                    
                    max_neuron_idx = np.max(spike_indices) if len(spike_indices) > 0 else 0
                    axes[i].set_ylim(-0.5, max_neuron_idx + 0.5)
            else:
                axes[i].text(0.5, 0.5, 'No spikes', ha='center', va='center', transform=axes[i].transAxes)
                axes[i].set_ylabel(f'{neuron_type}\nNeuron #')
                axes[i].set_ylim(-0.5, 0.5)
        else:
            axes[i].text(0.5, 0.5, 'No data', ha='center', va='center', transform=axes[i].transAxes)
            axes[i].set_ylabel(f'{neuron_type}\nNeuron #')
            axes[i].set_ylim(-0.5, 0.5)
    
    axes[-1].set_xlabel('Time (ms)')
    plt.tight_layout()
    plt.show()

def tune_network_parameters(params):
    
    for conn_name, conn_config in params['connections'].items():
        if 'Cortex_to_MSND' in conn_name:
            # MSN 입력 대폭 감소
            original_weight = conn_config['weight']
            conn_config['weight'] = 0.005  # 0.05 -> 0.005 (1/10)
            conn_config['p'] = 0.1  # 0.3/0.7 -> 0.1 (연결 확률 감소)
            print(f'   {conn_name}: weight {original_weight} -> {conn_config["weight"]}, p -> {conn_config["p"]}')
            
        elif 'Cortex_to_FSN' in conn_name:
            # FSN 입력 약간 감소 (10-20Hz 목표)
            original_weight = conn_config['weight']
            conn_config['weight'] = 1.0  # 2 -> 1.0
            print(f'   {conn_name}: weight {original_weight} -> {conn_config["weight"]}')
            
        elif 'Cortex_to_STN' in conn_name:
            # STN 입력 유지 또는 약간 감소
            original_weight = conn_config['weight']
            conn_config['weight'] = 0.3  # 0.5 -> 0.3
            print(f'   {conn_name}: weight {original_weight} -> {conn_config["weight"]}')
    
    # External input 튜닝 (GPe를 33Hz 정도로)
    for conn_name, conn_config in params['connections'].items():
        if 'Ext_to_GPeT1' in conn_name:
            # GPeT1 입력 조정
            original_weight = conn_config['weight']
            conn_config['weight'] = 0.15  # 0.08 -> 0.15 (증가)
            print(f'   {conn_name}: weight {original_weight} -> {conn_config["weight"]}')
            
        elif 'Ext_to_GPeTA' in conn_name:
            # GPeTA 입력 조정
            original_weight = conn_config['weight']
            conn_config['weight'] = 0.6  # 0.4 -> 0.6 (증가)
            print(f'   {conn_name}: weight {original_weight} -> {conn_config["weight"]}')
    
    return params

def main():
    print('⚡ 튜닝된 기저핵 네트워크 시뮬레이션')
    print('=' * 55)
    print('목표 발화율: MSN 0.01-0.2Hz, FSN/STN 10-20Hz, GPe 33Hz')
    
    start_time = time.time()
    
    # 기존 설정 로드
    params = load_params('config/test_normal_noin.json')
    
    # 중간 크기로 뉴런 개수 설정
    for neuron in params['neurons']:
        if neuron['name'] == 'Cortex_FSN':
            neuron['N'] = 20
        elif neuron['name'] == 'Cortex_MSND1':
            neuron['N'] = 200
        elif neuron['name'] == 'Cortex_MSND2':
            neuron['N'] = 200
        elif neuron['name'] == 'Cortex_STN':
            neuron['N'] = 5
        elif neuron['name'] == 'Ext_GPeT1':
            neuron['N'] = 15
        elif neuron['name'] == 'Ext_GPeTA':
            neuron['N'] = 10
        elif neuron['name'] == 'FSN':
            neuron['N'] = 20
        elif neuron['name'] == 'STN':
            neuron['N'] = 5
        elif neuron['name'] == 'MSND1':
            neuron['N'] = 200
        elif neuron['name'] == 'MSND2':
            neuron['N'] = 200
        elif neuron['name'] == 'GPeT1':
            neuron['N'] = 15
        elif neuron['name'] == 'GPeTA':
            neuron['N'] = 10
    
    # 파라미터 튜닝
    params = tune_network_parameters(params)
    
    # 시뮬레이션 시간
    params['simulation']['duration'] = 2000  # 2초 (더 정확한 발화율 측정)
    params['start_time'] = 0
    params['end_time'] = 2000
    
    print('\n🧠 뉴런 구성:')
    total_neurons = 0
    for neuron in params['neurons']:
        if 'model_class' in neuron:
            print(f'   - {neuron["name"]}: {neuron["N"]}개')
            total_neurons += neuron['N']
    
    print(f'📊 총 뉴런 수: {total_neurons}개')
    print(f'⏱️  시뮬레이션 시간: {params["simulation"]["duration"]}ms')
    
    # 외부 입력 설정
    ext_inputs = {}
    for neuron_config in params['neurons']:
        if neuron_config.get('neuron_type') == 'poisson':
            if 'target_rates' in neuron_config:
                target, rate_info = list(neuron_config['target_rates'].items())[0]
                rate_expr = rate_info['equation']
                ext_inputs[target] = {'rate': rate_expr}
    
    print('\n🚀 튜닝된 시뮬레이션 실행 중...')
    
    results = run_simulation_with_inh_ext_input(
        neuron_configs=params['neurons'],
        connections=params['connections'],
        synapse_class=params['synapse_class'],
        simulation_params=params['simulation'],
        plot_order=params['plot_order'],
        start_time=0*ms,
        end_time=2000*ms,
        ext_inputs=ext_inputs
    )
    
    end_time = time.time()
    
    print('\n' + '='*55)
    print('📊 튜닝 결과 분석')
    print('='*55)
    print(f'⏱️  실행 시간: {end_time - start_time:.2f}초')
    
    # 발화율 결과 및 목표 비교
    if 'firing_rates' in results:
        print('\n🎯 발화율 결과 vs 목표값:')
        target_rates = {
            'MSND1': (0.01, 0.2),
            'MSND2': (0.01, 0.2), 
            'FSN': (10, 20),
            'STN': (10, 20),
            'GPeT1': (30, 40),
            'GPeTA': (30, 40)
        }
        
        for neuron_type, rate in results['firing_rates'].items():
            if neuron_type in target_rates:
                min_target, max_target = target_rates[neuron_type]
                if min_target <= rate <= max_target:
                    status = "✅ 목표 달성"
                elif rate > 0:
                    status = "⚠️ 조정 필요"
                else:
                    status = "❌ 발화 없음"
                
                print(f'   {neuron_type}: {rate:.3f} Hz {status} (목표: {min_target}-{max_target} Hz)')
            else:
                print(f'   {neuron_type}: {rate:.3f} Hz')
        
        # 튜닝 제안
        print('\n💡 추가 튜닝 제안:')
        for neuron_type, rate in results['firing_rates'].items():
            if neuron_type in target_rates:
                min_target, max_target = target_rates[neuron_type]
                if rate > max_target:
                    if 'MSN' in neuron_type:
                        print(f'   {neuron_type}: Cortex weight를 더 줄이세요 (현재: {rate:.3f} > {max_target})')
                    elif neuron_type in ['GPeT1', 'GPeTA']:
                        print(f'   {neuron_type}: External weight를 줄이세요 (현재: {rate:.3f} > {max_target})')
                elif rate < min_target:
                    if 'MSN' in neuron_type:
                        print(f'   {neuron_type}: Cortex weight를 늘리세요 (현재: {rate:.3f} < {min_target})')
                    elif neuron_type in ['FSN', 'STN']:
                        print(f'   {neuron_type}: Cortex weight를 늘리세요 (현재: {rate:.3f} < {min_target})')
                    elif neuron_type in ['GPeT1', 'GPeTA']:
                        print(f'   {neuron_type}: External weight를 늘리세요 (현재: {rate:.3f} < {min_target})')
    
    # Raster plot 생성
    print('\n📈 Raster plot 생성 중...')
    try:
        plot_raster(results, params['plot_order'], duration_ms=2000)
    except Exception as e:
        print(f"Raster plot 생성 실패: {e}")
    
    print('\n✨ 튜닝된 기저핵 네트워크 시뮬레이션 완료!')
    
    return results

if __name__ == "__main__":
    main() 