from brian2 import *
import matplotlib.pyplot as plt

class Simulation:
    def __init__(self, neuron_model, synapse_model=None):
        self.neuron_model = neuron_model
        self.synapse_model = synapse_model
        self.setup_monitors()
        self.network = Network(self.neuron_model.neurons, self.dv_monitor, self.spike_monitor, self.rate_monitor)
        if self.synapse_model:
            self.network.add(self.synapse_model.synapses)

    def setup_monitors(self):
        self.dv_monitor = StateMonitor(self.neuron_model.neurons, 'v', record=True)
        self.spike_monitor = SpikeMonitor(self.neuron_model.neurons)
        self.rate_monitor = PopulationRateMonitor(self.neuron_model.neurons)
    
    def build_network(self):
        self.network = Network(self.neuron_model.neurons, self.dv_monitor, self.spike_monitor, self.rate_monitor)
        if self.synapse_model:
            self.network.add(self.synapse_model.synapses)

    def run(self, duration):
        self.network.run(duration)
        if plot:
            self.plot_results(earliest_time_stabilized)

    def plot_results(self, earliest_time_stabilized=None):
        plt.figure(figsize=(12, 8))

        # Plot membrane potential
        plt.subplot(3, 1, 1)
        plt.plot(self.dv_monitor.t / ms, self.dv_monitor.v[0] / mV, label='Membrane Potential')
        if earliest_time_stabilized:
            plt.axvline(x=earliest_time_stabilized / ms, color='gray', linestyle='--', label='Stabilization')
        plt.xlabel('Time (ms)')
        plt.ylabel('Membrane Potential (mV)')
        plt.legend()

        # Plot spikes
        plt.subplot(3, 1, 2)
        if len(self.spike_monitor.i) > 0:
            spike_times_of_neuron_0 = self.spike_monitor.t[self.spike_monitor.i == 0]
            plt.plot(spike_times_of_neuron_0 / ms, np.zeros_like(spike_times_of_neuron_0), 'o', markersize=2, label='Spikes (Neuron 0)')
        else:
            plt.text(0.5, 0.5, 'No spikes recorded', fontsize=12, ha='center', va='center')
        plt.xlabel('Time (ms)')
        plt.ylabel('Neuron Index (0)')
        plt.yticks([0])
        plt.legend()
        
        # Plot firing rate
        plt.subplot(3, 1, 3)
        if len(self.rate_monitor.t) > 0:
            plt.plot(self.rate_monitor.t / ms, self.rate_monitor.smooth_rate(width=10*ms) / Hz, label='Firing Rate')
        else:
            plt.text(0.5, 0.5, 'No firing rate data', fontsize=12, ha='center', va='center')
        plt.xlabel('Time (ms)')
        plt.ylabel('Firing Rate (Hz)')
        plt.legend()

        plt.tight_layout()
        plt.show()