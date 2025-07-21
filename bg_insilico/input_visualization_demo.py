#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
from brian2 import *
import random

# Font settings
plt.rcParams['font.family'] = 'DejaVu Sans'

def create_poisson_input_visualization():
    """
    Poisson 기반 입력이 뉴런 인덱스마다 겹치지 않게 들어오는 것을 시각화
    """
    
    # 시뮬레이션 파라미터
    duration = 30000  # ms (30초 실험)
    dt = 0.1  # ms
    n_neurons = 20  # 뉴런 수
    base_rate = 5.0  # Hz (기본 발화율)
    stimulus_rate = 50.0  # Hz (자극 시 발화율)
    
    # 시간 축
    time_vector = np.arange(0, duration, dt)
    
    # 자극 구간 설정 (10000-11000ms, 1초간)
    stim_start = 10000
    stim_end = 11000
    
    # 각 뉴런별 Poisson 스파이크 생성
    neuron_spikes = {}
    
    for neuron_idx in range(n_neurons):
        spikes = []
        
        for t in time_vector:
            # 자극 구간인지 확인
            if stim_start <= t <= stim_end:
                rate = stimulus_rate
            else:
                rate = base_rate
            
            # Poisson 확률로 스파이크 생성
            prob = rate * dt / 1000.0  # Hz를 확률로 변환
            if random.random() < prob:
                spikes.append(t)
        
        neuron_spikes[neuron_idx] = spikes
    
    # 시각화
    fig, ax = plt.subplots(1, 1, figsize=(16, 8))
    
    # Raster Plot (스파이크 시점)
    print("Creating raster plot...")
    for neuron_idx in range(n_neurons):
        spikes = neuron_spikes[neuron_idx]
        if spikes:
            ax.scatter(spikes, [neuron_idx] * len(spikes), 
                      s=20, alpha=0.9, color='gold', edgecolors='none')
    
    ax.set_xlabel('Time (seconds)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Neuron Index', fontsize=12, fontweight='bold')
    ax.set_title('Poisson Input Raster Plot\n(Independent spikes for each neuron)', fontsize=14, fontweight='bold')
    ax.set_xlim(0, duration)
    ax.set_ylim(-0.5, n_neurons - 0.5)
    ax.grid(True, alpha=0.3, axis='x')
    
    # X축 틱 간격 조정 (5초마다)
    ax.set_xticks(np.arange(0, duration + 1, 5000))
    ax.set_xticklabels([f'{int(t/1000)}s' for t in np.arange(0, duration + 1, 5000)])
    
    # 자극 구간 표시
    ax.axvspan(stim_start, stim_end, alpha=0.1, color='red', label='Stimulus Period')
    ax.legend()
    
    plt.tight_layout()
    
    # 파일 저장
    plt.savefig('poisson_input_visualization.png', dpi=300, bbox_inches='tight')
    print("Poisson input visualization saved to 'poisson_input_visualization.png'")
    
    # 화면에 표시
    plt.show()
    
    # 추가 통계 출력
    print(f"\n=== Poisson Input Statistics ===")
    print(f"Total spikes generated: {sum(len(spikes) for spikes in neuron_spikes.values())}")
    print(f"Average spikes per neuron: {np.mean([len(spikes) for spikes in neuron_spikes.values()]):.1f}")
    print(f"Spike rate during baseline: {base_rate} Hz")
    print(f"Spike rate during stimulus: {stimulus_rate} Hz")
    
    return neuron_spikes

def create_individual_neuron_input_demo():
    """
    개별 뉴런의 입력 패턴을 더 자세히 보여주는 시각화
    """
    
    # 시뮬레이션 파라미터
    duration = 500  # ms
    dt = 0.1  # ms
    n_neurons = 5  # 뉴런 수 (시각화를 위해 적게)
    
    # 각 뉴런별 다른 발화율 설정
    neuron_rates = {
        0: {'base': 3.0, 'stim': 40.0},   # Low firing rate neuron
        1: {'base': 8.0, 'stim': 60.0},   # Medium firing rate neuron
        2: {'base': 15.0, 'stim': 80.0},  # High firing rate neuron
        3: {'base': 5.0, 'stim': 50.0},   # Medium firing rate neuron
        4: {'base': 12.0, 'stim': 70.0}   # High firing rate neuron
    }
    
    # 자극 구간
    stim_start = 150
    stim_end = 350
    
    # 시간 축
    time_vector = np.arange(0, duration, dt)
    
    # 각 뉴런별 Poisson 스파이크 생성
    neuron_spikes = {}
    
    for neuron_idx in range(n_neurons):
        rates = neuron_rates[neuron_idx]
        spikes = []
        
        for t in time_vector:
            # 자극 구간인지 확인
            if stim_start <= t <= stim_end:
                rate = rates['stim']
            else:
                rate = rates['base']
            
            # Poisson 확률로 스파이크 생성
            prob = rate * dt / 1000.0
            if random.random() < prob:
                spikes.append(t)
        
        neuron_spikes[neuron_idx] = spikes
    
    # 시각화
    fig, axes = plt.subplots(n_neurons, 1, figsize=(12, 10))
    if n_neurons == 1:
        axes = [axes]
    
    for neuron_idx in range(n_neurons):
        ax = axes[neuron_idx]
        spikes = neuron_spikes[neuron_idx]
        rates = neuron_rates[neuron_idx]
        
        # 스파이크 표시
        if spikes:
            ax.scatter(spikes, [1] * len(spikes), s=30, alpha=0.9, 
                      color='gold', edgecolors='none')
        
        # 발화율 변화 표시
        time_bins = np.arange(0, duration + 20, 20)
        rate_data = []
        
        for t_start in time_bins[:-1]:
            t_end = t_start + 20
            spikes_in_bin = sum(1 for spike in spikes if t_start <= spike < t_end)
            rate = (spikes_in_bin / (20 / 1000.0))  # Hz
            rate_data.append(rate)
        
        time_centers = time_bins[:-1] + 10
        ax.plot(time_centers, rate_data, 'blue', linewidth=1.5, alpha=0.7)
        
        # 설정
        ax.set_xlim(0, duration)
        ax.set_ylim(0, max(rate_data) * 1.2 if rate_data else 10)
        ax.set_ylabel(f'N{neuron_idx}', fontsize=10, fontweight='bold')
        
        # 자극 구간 표시
        ax.axvspan(stim_start, stim_end, alpha=0.2, color='orange')
        
        if neuron_idx == n_neurons - 1:
            ax.set_xlabel('Time (ms)', fontsize=12, fontweight='bold')
        
        # 제목
        ax.set_title(f'Neuron {neuron_idx}: Base={rates["base"]}Hz, Stim={rates["stim"]}Hz', 
                    fontsize=11, fontweight='bold')
    
    plt.suptitle('Individual Neuron Poisson Input Patterns', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    # 파일 저장
    plt.savefig('individual_neuron_input_patterns.png', dpi=300, bbox_inches='tight')
    print("Individual neuron input patterns saved to 'individual_neuron_input_patterns.png'")
    
    plt.show()

if __name__ == "__main__":
    print("=== Poisson Input Visualization Demo ===")
    
    # 1. 기본 Poisson 입력 시각화
    print("\n1. Creating basic Poisson input visualization...")
    neuron_spikes = create_poisson_input_visualization()
    
    # 2. 개별 뉴런 입력 패턴 시각화 (비활성화)
    # print("\n2. Creating individual neuron input patterns...")
    # create_individual_neuron_input_demo()
    
    print("\n=== Demo Complete ===")
    print("Generated files:")
    print("- poisson_input_visualization.png")
    # print("- individual_neuron_input_patterns.png") 