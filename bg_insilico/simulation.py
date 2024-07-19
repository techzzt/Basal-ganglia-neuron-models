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
        self.plot_results()

    def plot_results(self):
        dv_dt = np.diff(self.dv_monitor.v, axis=1) / (self.dv_monitor.t[1] - self.dv_monitor.t[0])

        plt.figure(figsize=(12, 8))
        plt.subplot(3, 1, 1)
        plt.plot(self.dv_monitor.t[:-1] / ms, dv_dt.T / mV * 1000, lw=0.5)
        plt.xlabel('Time')
        plt.ylabel('dv/dt')
        plt.title('Membrane Potential Change (dv/dt)')
        plt.legend(['Neuron {}'.format(i) for i in range(10)], loc='upper right', fontsize='small', ncol=5)

        plt.subplot(3, 1, 2)
        plt.scatter(self.spike_monitor.t / ms, self.spike_monitor.i, s=2, c='red', label='Spikes')
        plt.xlabel('Time')
        plt.ylabel('Neuron index')
        plt.title('Spike Events')
        plt.legend()

        plt.subplot(3, 1, 3)
        plt.plot(self.rate_monitor.t / ms, self.rate_monitor.smooth_rate(width=10*ms) / Hz, label='Firing rate (Hz)')
        plt.xlabel('Time')
        plt.ylabel('Firing rate (Hz)')
        plt.title('Population Firing Rate')
        plt.legend()

        plt.tight_layout()
        plt.show()