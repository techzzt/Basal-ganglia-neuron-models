import matplotlib.pyplot as plt
import numpy as np
from brian2 import ms, pA
from brian2 import Network, StateMonitor, SpikeMonitor, PopulationRateMonitor


class Run:
    def __init__(self, neuron_model):
        self.neuron_model = neuron_model
        self.network = Network(neuron_model.neurons)
        
        # 기본 모니터 설정
        self.dv_monitor = StateMonitor(neuron_model.neurons, 'v', record=True)
        self.spike_monitor = SpikeMonitor(neuron_model.neurons)
        self.rate_monitor = PopulationRateMonitor(neuron_model.neurons)
        self.current_monitor = StateMonitor(neuron_model.neurons, 'I', record=True)
        
        self.network.add(self.dv_monitor, self.spike_monitor, self.rate_monitor, self.current_monitor)

        # 멤브레인 전위 및 전류를 저장할 리스트
        self.membrane_potentials = []
        self.input_currents = []

    def run(self, duration):
        self.network.run(duration)
        # 시뮬레이션 후에 멤브레인 전위와 입력 전류를 저장
        self.membrane_potentials.append(self.dv_monitor.v[0])
        self.input_currents.append(self.current_monitor.I[0])