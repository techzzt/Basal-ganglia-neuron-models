from brian2 import Network, StateMonitor, SpikeMonitor, PopulationRateMonitor
import matplotlib.pyplot as plt
import numpy as np
from brian2 import ms, mV, pA, pF, siemens, amp, second, Hz


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

    def run(self, duration):
        self.network.run(duration)
    
    def plot_results(self, earliest_time_stabilized=None):
        plt.figure(figsize=(15, 15))

        # Membrane potential
        plt.subplot(4, 1, 1)
        plt.plot(self.dv_monitor.t / ms, self.dv_monitor.v[0] / mV, label='Membrane Potential')
        if earliest_time_stabilized:
            plt.axvline(x=earliest_time_stabilized / ms, color='gray', linestyle='--', label='Stabilization')
        plt.xlabel('Time (ms)')
        plt.ylabel('Membrane Potential (mV)')
        plt.legend()

        # Spikes
        plt.subplot(4, 1, 2)
        spikes_of_neuron_0 = self.spike_monitor.i[self.spike_monitor.i == 0] 
        spike_times_of_neuron_0 = self.spike_monitor.t[self.spike_monitor.i == 0]  
        plt.plot(spike_times_of_neuron_0 / ms, spikes_of_neuron_0, '.', markersize=2, label='Spikes (Neuron 0)')
        plt.xlabel('Time (ms)')
        plt.ylabel('Neuron Index (0)')
        plt.yticks([0])  # y축에 0만 표시
        plt.legend()

        # Firing rate
        plt.subplot(4, 1, 3)
        plt.plot(self.rate_monitor.t / ms, self.rate_monitor.smooth_rate(width=10*ms) / Hz, label='Firing Rate')
        plt.xlabel('Time (ms)')
        plt.ylabel('Firing Rate (Hz)')
        plt.legend()

        # Input current
        plt.subplot(4, 1, 4)
        initial_time = np.arange(0, 1000, 1) 
        initial_current = np.zeros_like(initial_time)  
        total_time = np.concatenate((initial_time, self.current_monitor.t / ms))
        total_current = np.concatenate((initial_current, self.current_monitor.I[0] / pA))

        # Plot the current
        plt.plot(total_time, total_current, label='Current I (pA)', color='orange')
        plt.xlabel('Time (ms)')
        plt.ylabel('Current (pA)')
        plt.xlim(0, self.dv_monitor.t[-1] / ms)  # 수정된 부분
        plt.legend()

        plt.tight_layout()
        plt.show()
