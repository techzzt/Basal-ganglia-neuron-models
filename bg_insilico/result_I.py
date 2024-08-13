import matplotlib.pyplot as plt
import numpy as np
from brian2 import ms, pA
from brian2 import Network, StateMonitor, SpikeMonitor, PopulationRateMonitor


class Visualization:
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

    def plot_results(self):
        plt.figure(figsize=(15, 15))

        for i, (membrane_potential, input_current) in enumerate(zip(self.membrane_potentials, self.input_currents)):
            plt.subplot(10, 2, i*2 + 1)
            plt.plot(self.dv_monitor.t / ms, membrane_potential / mV, label=f'Membrane Potential (I = {-90 + 30*i} pA)')
            plt.xlabel('Time (ms)')
            plt.ylabel('Membrane Potential (mV)')
            plt.legend()

            plt.subplot(10, 2, i*2 + 2)
            plt.plot(self.current_monitor.t / ms, input_current / pA, label=f'Input Current (I = {-90 + 30*i} pA)', color='orange')
            plt.xlabel('Time (ms)')
            plt.ylabel('Current (pA)')
            plt.legend()

        plt.tight_layout()
        plt.show()
